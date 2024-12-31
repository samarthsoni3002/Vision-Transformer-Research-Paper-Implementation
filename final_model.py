import torch
import torch.nn as nn
from image_embedding import ImageEmbedding
from vision_transformer import VisionTransformer



class FinalModel(nn.Module):
    
    def __init__(self,
                 img_size = 224,
                 in_channels = 3,
                 patch_size = 16,
                 num_transformer_layers=12,
                 embedding_dim = 768,
                 mlp_size=3072,
                 num_head=12,
                 attention_dropout=0,
                 mlp_dropout=0.1,
                 embedding_dropout=0.1,
                 num_classes=120,
                 batch_size=32):
        
        
        super().__init__()
        
        assert img_size%patch_size == 0
        
        self.num_patches = int((img_size*img_size) // (patch_size*patch_size))
        
        self.image_embedding = ImageEmbedding(number_of_patches=self.num_patches,
                                              batch_size=batch_size)
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.vision_transformer = nn.Sequential(*[VisionTransformer(embedding_dim=embedding_dim,
                                num_heads=num_head,
                                mlp_dropout=mlp_dropout,
                                attention_dropout=attention_dropout,
                                mlp_size=3072) for _ in range(num_transformer_layers)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    
    def forward(self,x):
        
        x = self.image_embedding(x)
        x = self.embedding_dropout(x)
        x = self.vision_transformer(x)
        x = self.classifier(x[:,0])
        return x
    
    


