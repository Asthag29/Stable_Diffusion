import torch
from torch import nn
from torch.nn import functional as F
from vae_block import Vae_ResidualBlock
from vae_block import Vae_AttentionBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):

        super().__init__(

            nn.Conv2d(3, 128 , kernel_size=3 , padding = 1),               # batch_size , 3 , height , width -> (batch_size , 128  , height , width) 
            Vae_ResidualBlock(128, 128 ),   #(batch_size, 128 , height , width) --> (batch_size , 128 , height , width)
            Vae_ResidualBlock(128, 128),    #(batch_size, 128 , height , width) --> (batch_size , 128 , height , width)

            nn.Conv2d(128, 128 , kernel_size=3 , stride = 2, padding = 0),  # (batch_size , 128 , height , width) -> (batch_size , 128 , height/2 , width/2)
            Vae_ResidualBlock(128, 256 ),  # (batch_size , 128 , height/2 , width/2) -> (batch_size , 256 , height/2 , width/2)
            Vae_ResidualBlock(256, 256 ),  # (batch_size , 256 , height/2 , width/2) -> (batch_size , 256 , height/2 , width/2)

            nn.Conv2d(256, 256 , kernel_size=3 , stride = 2, padding = 0),  # (batch_size , 256 , height/2 , width/2) -> (batch_size , 256 , height/4 , width/4)
            Vae_ResidualBlock(256, 512 ),  # (batch_size , 256 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            Vae_ResidualBlock(512, 512 ),  # (batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)

            nn.Conv2d(512, 512, kernel_size=3 , stride = 2, padding = 0),  # (batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512 ),   #(batch_size , 512 , height/8 , width/8) -> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512 ),   #(batch_size , 512 , height/8 , width/8) -> (batch_size , 512 , height/8 , width/8)

            Vae_ResidualBlock(512, 512 ),   #(batch_size , 512 , height/8 , width/8) -> (batch_size , 512 , height/8 , width/8)
            Vae_AttentionBlock(512),        #(batch_size , 512 , height/8 , width/8) -> (batch_size , 512 , height/8 , width/8) 
            Vae_ResidualBlock(512, 512),     #(batch_size , 512 , height/8 , width/8) -> (batch_size , 512 , height/8 , width/8)

            nn.GroupNorm(32, 512),         

            nn.SiLU(), 

            nn.Conv2d(512 , 8 , kernel_size = 3 , padding = 1), #  (batch_size , 512 , height/8 , width/8) -> (batch_size , 8 , height/8 , width/8)

            nn.Conv2d(8 , 8 , kernel_size = 1 , padding = 0),   # (batch_size , 8 , height/8 , width/8) -> (batch_size , 8 , height/8 , width/8)

        )

    def forward(self, x: torch.Tensor , noise: torch.Tensor):   # data(x), noise = (batchsize , channel , height , width)

        for module in self:
            if getattr(module , "stride", None ) == (2,2):
                x = F.pad(x , (0,1,0,1)) 
            x = module(x)

            
        mean , log_variance = torch.chunk(x , 2 , dim=1)    #(batch_size , 8 , height/8 , width/8) --> 2*  (batch_size , 4 , height/8 , width/8)

        log_variance = torch.clamp(log_variance , -30 , 20)  #clamping all the values between -30 and 20 for stability(all values less than -30 are set to -30 and all values greater than 20 are set to 20)
        variance = log_variance.exp()
        stdev= variance.sqrt()
        
        x = mean + stdev * noise    # (batch_size , 4 , height/8 , width/8)
        x *= 0.8214     #scaling factor for stability(no idea where this number came from)
        return x
