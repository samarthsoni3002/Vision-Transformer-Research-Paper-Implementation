import torch
import torch.nn as nn



class MLPBlock(nn.Module):
    
    def __init__(self,
                 embedding_dim: int,
                 mlp_size: int,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
        
        self.layer1 = nn.Linear(in_features=embedding_dim,
                                out_features=mlp_size)
        
        self.layer2 = nn.Linear(in_features=mlp_size,out_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        
    def forward(self,x):
        
        x = self.layer_norm(x)
        x = self.gelu(self.layer1(x))
        x = self.layer2(self.dropout(x))
        
        return self.dropout(x)
    
    
    
    
    