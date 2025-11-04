from torch import nn
from .vae_block import Vae_AttentionBlock
from .vae_block import Vae_ResidualBlock
    

class VAE_Decoder(nn.Sequential):

    def __init__(self):

        super().__init__(
            nn.Conv2d(4, 4 , kernel_size=1 , padding = 0),        #(batch_size, 4 , height/8 , width/8) --> (batch_size , 4 , height/8 , width/8) , padding = 1
            nn.Conv2d(4 , 512 , kernel_size = 3 , padding = 1),     #(batch_size, 4 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8) , padding = 1


            Vae_ResidualBlock(512, 512 ),        #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)
            Vae_AttentionBlock(512),            #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512),        #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512 ),       #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512),       #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)
            Vae_ResidualBlock(512, 512 ),      #(batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/8 , width/8)

            # upsampling by taking the value from the nearest neighbour
            nn.Upsample(scale_factor=2),        # (batch_size, 512 , height/8 , width/8) --> (batch_size , 512 , height/4 , width/4)

            nn.Conv2d(512, 512 , kernel_size=3 ,padding = 1),   # (batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            Vae_ResidualBlock(512, 512 ),                   #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            Vae_ResidualBlock(512, 512 ),                   #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            Vae_ResidualBlock(512, 512 ),                   #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)

            # nn.Conv2d(512, 512, kernel_size=3 , padding = 0),   #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/2 , width/2)
            # Vae_ResidualBlock(512, 512 ),                    #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            # Vae_ResidualBlock(512, 512 ),                       #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)
            # Vae_ResidualBlock(512, 512 ),            #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/4 , width/4)

            nn.Upsample(scale_factor=2),        #(batch_size , 512 , height/4 , width/4) -> (batch_size , 512 , height/2 , width/2)

            nn.Conv2d(512, 512, kernel_size=3 , padding = 1),   #(batch_size , 512 , height/2 , width/2) -> (batch_size , 512 , height/2 , width/2)
            # Vae_ResidualBlock(512, 512 ),                    #(batch_size , 512 , height/2 , width/2) -> (batch_size , 512 , height/2 , width/2)
            # Vae_ResidualBlock(512, 512 ),                       #(batch_size , 512 , height/2 , width/2) -> (batch_size , 512   , height/2 , width/2)       
            Vae_ResidualBlock(512, 256 ),       #(batch_size , 512 , height/2 , width/2) -> (batch_size , 256 , height/2 , width/2)
            Vae_ResidualBlock(256, 256 ),       #(batch_size , 256 , height/2 , width/2) -> (batch_size , 256 , height/2 , width/2)
            Vae_ResidualBlock(256, 256 ),       #(batch_size , 256 , height/2 , width/2) -> (batch_size , 256 , height/2 , width/2)

            nn.Upsample(scale_factor=2),     #(batch_size , 256 , height/2 , width/2) -> (batch_size , 256 , height , width)

            nn.Conv2d(256, 256, kernel_size=3 , padding = 1),   #(batch_size , 256 , height , width) -> (batch_size , 256 , height , width)
            Vae_ResidualBlock(256 ,128),        #(batch_size , 256 , height , width) -> (batch_size , 128 , height , width)
            Vae_ResidualBlock(128, 128 ),       #(batch_size , 128 , height , width) -> (batch_size , 128 , height , width)
            Vae_ResidualBlock(128, 128 ),       #(batch_size , 128 , height , width) -> (batch_size , 128 , height , width)

            nn.GroupNorm(32, 128),      # divide 128 features in group of 32

            nn.SiLU(),

            nn.Conv2d(128 , 3 , kernel_size=3 , padding=1)    # (batch_size , 128 , height , width) -> (batch_size , 3 , height , width)

        )

    def forward(self , x):
            
            x /= 0.18214    #dividing by the same number which was multiplied in the encoder for stability

            for module in self:
                 x= module(x)

            return x
    