import torch 
import torch.nn as nn
from MHA_block import MultiHeadAttention
from MLP_block import MLPBlock


class VisionTransformer(nn.Module):
    
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_size: int,
                 mlp_dropout: int,
                 attention_dropout: int):
        
        super().__init__()
        
        self.mha = MultiHeadAttention(num_head=num_heads,
                                           embedding_dim=embedding_dim,
                                           attention_dropout=attention_dropout)
        
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size)
        
        
        
        
    def forward(self,x):
        x = self.mha(x) + x
        x = self.mlp_block(x) + x
        
        return x