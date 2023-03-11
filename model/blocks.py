"""
This file carries the model blocks for the transformer model.
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  """One head of self-attention"""
  def __init__(self, n_embd, head_size=32, block_size=128, dropout=0.1, encoder=True):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    
    # mask buffer is not a parameter of the model, but is still saved with the model
    if encoder: # attend to all positions in the sequence
      self.register_buffer('mask', torch.ones(block_size, block_size))
    else: # attend to all positions before the current one (decoder)
      self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix
    
    self.dropout = nn.Dropout(dropout)
        
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   # what we have
    q = self.query(x) # what we want
    
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C**-0.5 # --> (B, T, T), dividing by sqrt(C) is a normalization trick
    wei = wei.masked_fill(self.mask[:T,:T]==0, float('-inf')) # masking the upper triangle
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    
    # weighted aggregation
    v = self.value(x) # selects what to focus on
    out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ Multiple Heads of self-attention in parallel """
  
  def __init__(self, num_heads, head_size, n_embd, dropout=0.1):
    super().__init__()
    # nn.ModuleList is a special container that registers the modules in the list
    #   "ModuleList can be indexed like a regular Python list, but modules it contains 
    #   are properly registered, and will be visible by all Module methods." - PyTorch docs
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads * head_size, n_embd) # to combine the heads smartly
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    # run each head in parallel then sum the results
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out
  
class FeedForward(nn.Module):
  """ a simple linear layer followed by non-linearity"""  
  def __init__(self, n_embd=32, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd), # projection layer that maps back to the original dimension
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.net(x)

class DecoderBlock(nn.Module):
  """Transformer block: communication via self-attention followed by computation via feed-forward"""
  
  def __init__(self, n_embd, n_head) -> None:
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    
  def forward(self, x):
    # layer norm done BEFORE the transformation (prenorm formulation) 
    # --> this is a deviation from the original paper which did it postnorm
    x = x + self.sa(self.ln1(x))  # x + is the residual connection
    x = x + self.ffwd(self.ln2(x))
    return x