from os import path

import tiktoken
from tiktoken import Encoding
import torch
from torch import Tensor
from torch.nn import functional as F

from model.gpt2_config import GPTConfig
from src.model.gpt2_nano import NanoGpt2


if __name__ == "__main__":
    
    # get device - cpu or cuda
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}")
    
    # load raw data Tiny Shakespeare
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    file_name = path.join('data', 'input.txt')
    with open (file_name, 'r') as datafile:
        raw_data = datafile.read()
    
    raw_data = raw_data[:1000]
    
    # get encoding
    tt_encoding: Encoding = tiktoken.get_encoding('gpt2')
    tokens: list[int] = tt_encoding.encode(raw_data)
    B, T = 4, 32
    token_buffer = torch.tensor(tokens[:B*T+1])
    # train data
    x = token_buffer[:-1].view(B,T)
    y = token_buffer[1:].view(B,T)
    # get our NanoGpt2 model with config for training
    
    model = NanoGpt2(GPTConfig())
    model.to(device)
    logits, loss = model(x, y)
    
    print(loss)
    import sys; sys.exit()
        
    # sequence data
    num_return_sequences: int = 5
    max_length: int = 30
    
    model.eval()

    # create input tokens
    tt_encoding: Encoding = tiktoken.get_encoding('gpt2')
    tokens: list[int] = tt_encoding.encode("Hello, I'm a language model,")
    tokens: Tensor = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens: Tensor = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    print("Shape of token tensor = ", tokens.shape)
    print("Token tensor = ", tokens)
    x: Tensor = tokens.to(device)

    # generate! right now x is (B, T) where B = 5, T = 8
    # append output tokens up to max_length
    # set the seed to 42
    torch.manual_seed(42)
    #torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        print(x.size(1))
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            logits = logits[0] # NEW convert from tuple to torch.Tensor
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs: Tensor = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix: Tensor = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens: list[int] = x[i, :max_length].tolist()
        decoded_text: str = tt_encoding.decode(tokens)
        print(">", decoded_text)
