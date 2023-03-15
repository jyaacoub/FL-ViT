"""
This file carries the model blocks for the transformer model.

Target transformer is Vit-B/16:
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
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
  def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
    super().__init__()
    head_size = hidden_dim // num_heads
    self.heads = nn.ModuleList([Head(hidden_dim, head_size) for _ in range(num_heads)])
    # combine heads by projecting to embedding dimension 
    self.proj = nn.Linear(num_heads * head_size, hidden_dim)
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
  def __init__(self, n_head=12, hidden_dim=768, mlp_dim=3072, 
               dropout=0.1, attn_dropout=0.1) -> None:
    super().__init__()
    head_size = hidden_dim // n_head
    self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6) # eps is for numerical stability
    
    self.sa = MultiHeadAttention(n_head, head_size, attn_dropout) # TODO: might be better to use nn.MultiheadAttention for speedups?
    self.dropout = nn.Dropout(dropout)
    
    self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)
    self.mlp = MlpBlock(hidden_dim, mlp_dim, dropout=dropout)
    
    
  def forward(self, inputs):
    # layer norm done BEFORE the transformation (prenorm formulation)
    x = self.ln1(inputs)
    x = self.sa(x)
    x = self.dropout(x)
    x = x + inputs # residual connection
    
    # mlp block
    y = self.ln2(x)
    y = self.mlp(y)
    return x + y # residual connection

class Encoder(nn.Module):
  """Encoder for the Vision Transformer"""
  def __init__(self, seq_len, n_layers, n_head=12, hidden_dim=768, mlp_dim=3072, 
               dropout=0.1, attn_dropout=0.1) -> None:
    super().__init__()
    
    # Learnable positional embedding instead of 
    # fixed positional encoding of the original Attention is all you need paper
    self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, hidden_dim).normal_(std=0.02))
    
    self.dropout = nn.Dropout(dropout)
    self.encoders = nn.Sequential(
      *[EncoderBlock(n_head, hidden_dim, mlp_dim, dropout, attn_dropout) for _ in range(n_layers)])
    self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
  
  def forward(self, x):
    x = x + self.pos_embedding
    x = self.dropout(x)
    return self.ln(self.encoders(x))

class VisionTransformer(nn.Module):
  """
  Vision Transformer from https://arxiv.org/abs/2010.11929:
  
  Default values are according to ViT-B/16.
  """
  def __init__(self, image_size=224, 
               patch_size=16,
               num_encoder_layers=12,
               hidden_dim=768,
               mlp_dim=3072, # 768*4 = 3072
               num_heads=12,
               dropout=0.1,
               attn_dropout=0.1) -> None:
    super().__init__()
    self.img_size = image_size
    self.patch_size = patch_size
    
    self.num_encoder_layers = num_encoder_layers
    
    self.hidden_dim = hidden_dim
    self.mlp_dim = mlp_dim
    
    self.num_heads = num_heads
    
    seq_len = (image_size // patch_size) ** 2 # 224/16 = 14 --> 14*14 = 196
    self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
    seq_len += 1 # add class token
    
    self.encoder = Encoder(seq_len, num_encoder_layers, 
                           num_heads, hidden_dim, 
                           mlp_dim, dropout, attn_dropout)
    
    # TODO: add classification head
    
    