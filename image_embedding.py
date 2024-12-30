import torch
import torch.nn as nn



class ImageEmbedding(nn.Module):
    
    def __init__(self,
                 in_channels=3,
                 out_channels=768,
                 patch_size = 16,
                 embedding_dim=768,
                 number_of_patches=196,
                 batch_size=32):
        
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)
        
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
        self.class_token = nn.Parameter(torch.ones(batch_size,1,embedding_dim),requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.ones(1,number_of_patches+1,embedding_dim), requires_grad=True)

         
    def forward(self,x):
        print(x.shape)
        
        if (x.shape[0] == 32):
            image_resolution = x.shape[-1]
            assert image_resolution % self.patch_size == 0
            
            x_patched = self.patcher(x)
            x_flatten = self.flatten(x_patched)
            x_flatten_permuted = x_flatten.permute(0,2,1)
            print(x_flatten_permuted.shape)
            print(self.class_token.shape)
            x_with_class_token = torch.cat((self.class_token,x_flatten_permuted),dim=1)
            return x_with_class_token + self.positional_embedding
        
    

