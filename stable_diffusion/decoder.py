import torch
from torch import nn
from torch.nn import functional as F
import math
from attention import Vae_AttentionBlock


# decoder residual block 
class Vae_ResidualBlock():
    def __init__(self, in_channels , out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels , out_channels, kernal_size=3 , padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(in_channels , out_channels, kernal_size=3 , padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else :
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        #x = (batchsize , in_channels , height , width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x= self.conv_1(x)

        x= self.groupnorm_2(x)

        x= F.selu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    
#decoder 

class VAE_Decoder(nn.Sequential):

    def __init__(self,x):

        super().__init__(
            nn.Conv2d(4, 4 , kernal_size=1 , padding = 0),        #(batch_size, channel , height , width) --> (batch_size , 128 , height , width) , padding = 1 
            nn.Conv2d(4 , 512 , kernal_size = 3 , padding = 1), 


            Vae_ResidualBlock(512, 512 ),
            Vae_AttentionBlock(512),
            Vae_ResidualBlock(512, 512),
            Vae_ResidualBlock(512, 512 ),   
            Vae_ResidualBlock(512, 512),   
            Vae_ResidualBlock(512, 512 ),


            nn.Upsample(scale_factor=2),



            nn.Conv2d(512, 512 , kernal_size=3 ,padding = 1),

            Vae_ResidualBlock(512, 512 ), 
            Vae_ResidualBlock(512, 512 ), 
            
            
            Vae_ResidualBlock(512, 512 ),

            nn.Conv2d(512, 512, kernal_size=3 , stride = 2, padding = 0),  
            Vae_ResidualBlock(512, 512 ),
            Vae_ResidualBlock(512, 512 ),
            Vae_ResidualBlock(512, 512 ),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernal_size=3 , padding = 1),

            Vae_ResidualBlock(512, 256 ),
            Vae_ResidualBlock(256, 256 ),
            Vae_ResidualBlock(256, 256 ),
            
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 52562, kernal_size=3 , padding = 1),

            Vae_ResidualBlock(256 ,128),
            Vae_ResidualBlock(128, 128 ),
            Vae_ResidualBlock(128, 128 ),

            nn.GroupNorm(32, 128),      # divide 128 features in group of 32

            nn.SiLU(),

            nn.Conv2d(128 , 3 , kernel_size=3 , padding=1)

        )

    def forward(self , x):
            
            x /= 0.18215

            for module in self:
                 x= module(x)

            return x
            









            





        

        


            
    