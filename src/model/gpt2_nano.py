import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.gpt2_building_blocks import DecoderBlock
from model.gpt2_config import GPTConfig


class NanoGpt2(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config: GPTConfig = config

        self.transformer = nn.ModuleDict(dict(
            # variables are named according to 
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            std = 0.02
            
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        # idx is of shape (B, T) (batch, time)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
            
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        
        # add token embeddings and positon embedding
        x = tok_emb + pos_emb
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            
        # forward the final layernorm 
        x = self.transformer.ln_f(x)
        
        # forward linear layer (classifier) and get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # get loss
        loss = None
        if targets is not None:
            loss: torch.Tensor = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024

        # create a from-scratch initialized nanoGPT model
        nano_gpt2_config: GPTConfig = GPTConfig(**config_args)
        nano_gpt2_model = NanoGpt2(nano_gpt2_config)
        sd = nano_gpt2_model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return nano_gpt2_model

    """
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {
                  len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {
                  len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(
            0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    """
 