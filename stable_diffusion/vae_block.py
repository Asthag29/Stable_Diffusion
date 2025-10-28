from attention import SelfAttention
import torch
from torch import nn
from torch.nn import functional as F


# Attention block 
class Vae_AttentionBlock(nn.Module):
    
    def __init__(self , channels: int):

        super().__init__()

        self.groupnorm = nn.GroupNorm(32 , channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):

        residue = x     #residue, x = (batchsize , channels , height , width)

        x = self.groupnorm(x)
        n , c , h , w = x.shape

        x = x.view(n , c , h*w)    # (batch_size , channel , height ,width) -> (batch_size  , channel , height * width)
        x = x.transpose(-1,-2)      # (batch_size  , channel , height * width) -> (batch_size , height * width , channel)

        x = self.attention(x)   # (batch_size , sequence/height*width , channel/embd) -> (batch_size , height * width , channel)

        x = x.transpose(-1, -2)     # (batch_size , height * width , channel) -> (batch_size  , channel , height * width)
        x = x.view(n,c,h,w)     # (batch_size  , channel , height * width) -> (batch_size , channel , height ,width)

        x += residue    #(batchsize , channels , height , width)

        return x
        

# Residual Block
class Vae_ResidualBlock(nn.Module):
    def __init__(self, in_channels , out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels , out_channels, kernel_size=3 , padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3 , padding=1)

        #skip connection(smooth gradient flow)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else :
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        
        residue = x             # residue = (batchsize , in_channels , height , width), x = (batchsize , in_channels , height , width)
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)
