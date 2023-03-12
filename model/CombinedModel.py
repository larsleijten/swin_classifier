import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, swin_encoder, mlp_head):
        super().__init__()
        self.swin_encoder = swin_encoder
        self.mlp_head = mlp_head
        
    def forward(self, x):
        x = self.swin_encoder(x)
        x = self.mlp_head(x)
        return x

