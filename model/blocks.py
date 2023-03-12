"""
This file carries the model blocks for the transformer model.
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  """
  Encoder head of self-attention (aka non-masked attention)
  """
  def __init__(self, n_embd, head_size=32, dropout=0.1):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)    
    self.dropout = nn.Dropout(dropout)
        
  def forward(self, x):
    B,T,C = x.shape   # batch, time, channels
    k = self.key(x)   # what we have
    q = self.query(x) # what we want
    
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C**-0.5 # --> (B, T, T), dividing by sqrt(C) is a normalization trick 
    # so that there is no sharping effect caused by SoftMax where if the values we are running through 
    # the SoftMax come from a larger domain they will converge to a one-hot vector
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    
    # weighted aggregation
    v = self.value(x) # selects what to focus on
    out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ Multiple Heads of self-attention in parallel """
  def __init__(self, num_heads, head_size, embd_dim, dropout=0.1):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    # combine heads by projecting to embedding dimension 
    self.proj = nn.Linear(num_heads * head_size, embd_dim)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    # run each head in parallel then concatenate the results
    out = torch.cat([h(x) for h in self.heads], dim=-1) # H,W,C --> H,W,C*num_heads
    out = self.proj(out) # H,W,C*num_heads --> H,W,C
    out = self.dropout(out)
    return out

class MlpBlock(nn.Module):
  """MLP block with GELU"""
  def __init__(self, D, emd_D, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(D, emd_D),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(emd_D, D),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.net(x)

class EncoderBlock(nn.Module):
  """Single ViT encoder block"""
  def __init__(self, n_embd, n_head, dropout=0.1) -> None:
    super().__init__()
    head_size = n_embd // n_head
    self.ln1 = nn.LayerNorm(n_embd, eps=1e-6) #eps is a small value added to the denominator to improve numerical stability
    
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ln2 = nn.LayerNorm(n_embd, eps=1e-6)
    
    self.mlp = MlpBlock(n_embd, dropout=dropout)
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, inputs):
    # layer norm done BEFORE the transformation (prenorm formulation)
    x = self.ln1(inputs)
    x = self.sa(x)
    x = self.dropout(x)
    x = x + inputs # residual connection
    
    # mlp block
    x = self.mlp(self.ln2(x)) + x
    return x