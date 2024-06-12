import tiktoken
from tiktoken import Encoding
import torch
from torch import Tensor
from torch.nn import functional as F

from src.model.gpt2_nano import GPT


if __name__ == "__main__":
    
    num_return_sequences: int = 5
    max_length: int = 30

    # get our GPT model
    model = GPT.from_pretrained('gpt2')
    model.eval()
    # model.to('cuda')

    # prefix tokens
    tt_encoding: Encoding = tiktoken.get_encoding('gpt2')
    tokens: list[int] = tt_encoding.encode("Hello, I'm a language model,")
    tokens: Tensor = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    # x = tokens.to('cuda')
    x = tokens

    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    #torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            logits = logits[0] # NEW convert from tuple to torch.Tensor
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
