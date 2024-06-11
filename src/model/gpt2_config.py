from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
