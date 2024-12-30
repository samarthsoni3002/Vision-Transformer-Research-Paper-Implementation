import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    
    def __init__(self,
                 num_head: int,
                 embedding_dim: int,
                 attention_dropout: float):
        
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=num_head,
                                                         dropout=attention_dropout,
                                                         batch_first=True)
        
        
    def forward(self, x):
        
        x = self.layer_norm(x)
        x,_ = self.multihead_attention(query=x,
                                       key=x,
                                       value=x,
                                       need_weights=False)
        
        return x