import torch
from torch import nn
from torch.nn import functional as F
import math

from decoder import Vae_AttentionBlock, Vae_ResidualBlock




class VAE_Encoder(nn.Sequential):

    def __init__(self,x):

        super().__init__(
            nn.Conv2d(3, 128 , kernal_size=3 , padding = 1),        #(batch_size, channel , height , width) --> (batch_size , 128 , height , width) , padding = 1 
            Vae_ResidualBlock(128, 128 ),   #(batch_size, 128 , height , width) --> (batch_size , 128 , height , width)
            Vae_ResidualBlock(128, 128),    #(batch_size, 128 , height , width) --> (batch_size , 128 , height , width)

            nn.Conv2d(3, 128 , kernal_size=3 , stride = 2, padding = 0),  
            Vae_ResidualBlock(128, 256 ), 
            Vae_ResidualBlock(256, 256 ), 

            nn.Conv2d(256, 256 , kernal_size=3 , stride = 2, padding = 0),  
            Vae_ResidualBlock(256, 512 ),
            Vae_ResidualBlock(512, 512 ),

            nn.Conv2d(512, 512, kernal_size=3 , stride = 2, padding = 0),  
            Vae_ResidualBlock(512, 512 ),
            Vae_ResidualBlock(512, 512 ),

            Vae_ResidualBlock(512, 512 ),
            Vae_AttentionBlock(512),
            Vae_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(), 

            nn.Conv2d(512 , 8 , kernal_size = 3 , padding = 1),

            nn.Conv2d(8 , 8 , kernal_size = 1 , padding = 0),     
            
            torch.chunk(x, 2, dim=1)  #why do we need this
        )

    def forward(self, x: torch.Tensor , noise: torch.Tensor):
        # x --> data (batch_size, channel , height , width)
        # noise --> data (batch_size, channel , height , width)

        for module in self:
            if getattr(module , "stride", None ) == (2,2):
                #(paddig left , right , top , bottom)
                x = F.pad(x , (0,1,0,1)) #why only right and bottom padding
            x = module(x)
        log_variance = torch.clamp(log_variance , -30 , 20)
        mean , log_variance = torch.chunk(x , 2 , dim=1)    #chunk divides the dim=1 (channel) into 2 tensor , so each with half the number of channel as before
        variance = log_variance.exp() 

        stdev= variance.sqrt()
        
        x = mean + stdev * noise
        x *= 0.8214
