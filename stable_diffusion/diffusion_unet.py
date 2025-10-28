import torch
from torch import nn
from torch.nn import functional as F
from diffusion_block import UNet_ResidualBlock , UNet_AttentionBlock



# simple neural network
class TimeEmbedding(nn.Module):     #current time step information 

    def __init__(self, n_embd):

        super().__init__()

        self.linear_1 = nn.Linear(n_embd , 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd , n_embd)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x

# upsampling block   
class Upsample(nn.Module):

    def __init__(self, channels):

        super().__init__()
        self.conv = nn.Conv2d(channels, channels , kernel_size=3, padding=1)

    def forward(self,x):

        x = F.interpolate(x , scale_factor=2 , mode="nearest")

        return self.conv(x)



class SwitchSequential(nn.Sequential):

    def forward(self, x , context , time):
        
        for layer in self:

            if isinstance(layer , UNet_AttentionBlock):     # input x and clip context vector
                x = layer(x , context)

            elif isinstance(layer , UNet_ResidualBlock):
                x = layer(x, time)

            else:
                x = layer(x)
                
        return x


    
class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        #(batch_size , 4 , height/8 , width/8) --> output from the vae model encoder
        self.encoders = nn.ModuleList([

            # (batch_size , 4 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8)
            SwitchSequential(nn.Conv2d(in_channels=4 , out_channels=320 , kernel_size=3 , padding=1)),
            # (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8)
            SwitchSequential(UNet_ResidualBlock(in_channels=320 , out_channels=320), UNet_AttentionBlock(n_heads=8, n_embd=40)),    #320 = 8*40
            # (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8)
            SwitchSequential(UNet_ResidualBlock(in_channels=320 , out_channels=320), UNet_AttentionBlock(n_heads=8, n_embd=40)),


            #(batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/16 , width/16)
            SwitchSequential(nn.Conv2d(in_channels=320, out_channels=320 , kernel_size=3, stride=2 , padding=1)),
            # (batch_size , 320 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16)
            SwitchSequential(UNet_ResidualBlock(in_channels=320 , out_channels=640), UNet_AttentionBlock(n_heads=8, n_embd=80)),    #640 = 8*80
            # (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16)
            SwitchSequential(UNet_ResidualBlock(in_channels=640 , out_channels=640), UNet_AttentionBlock(n_heads=8, n_embd=80)),    #640 = 8*80


            #(batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/32 , width/32)
            SwitchSequential(nn.Conv2d(640, 640 , kernel_size=3, stride=2 , padding=1)),
            # (batch_size , 640 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32)
            SwitchSequential(UNet_ResidualBlock( 640, 1280), UNet_AttentionBlock(8,160)),   #1280 = 8*160
            # (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32)
            SwitchSequential(UNet_ResidualBlock(128 , 1280), UNet_AttentionBlock(8,160)),   #1280 = 8*160


            #(batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(nn.Conv2d(1280, 1280 , kernel_size=3, stride=2 , padding=1)),
            # (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(UNet_ResidualBlock(1280 , 1280)),
            # (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(UNet_ResidualBlock(1280 , 1280)),

        ])
        self.bottleneck = SwitchSequential(
            # (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            UNet_ResidualBlock(1280,1280),
            
            # (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            UNet_AttentionBlock(8,160),     #1280 = 8*160

            # (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            UNet_ResidualBlock(1280,1280),
        )

        self.decoders = nn.ModuleList([

            
            #(batch_size , 2560 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(UNet_ResidualBlock(2560 , 1280)),  
            #(batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(UNet_ResidualBlock(2560 , 1280)),


            #(batch_size , 2560 , height/64 , width/64) --> (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/32 , width/32)
            SwitchSequential(UNet_ResidualBlock(2560 , 1280), Upsample(1280)),
            #(batch_size , 2560 , height/32 , width/32) --> (batch_size , 1280 , height/64 , width/64) --> (batch_size , 1280 , height/32 , width/32)
            SwitchSequential(UNet_ResidualBlock(2560 , 1280), UNet_AttentionBlock(8,160)),  #1280 = 8*160
            #(batch_size , 2560 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8,160)),

            #(batch_size , 1920 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/32 , width/32) --> (batch_size , 1280 , height/16 , width/16)
            SwitchSequential(UNet_ResidualBlock(1920 , 1280), UNet_AttentionBlock(8,160), Upsample(1280)),  #1280 = 8*160
            #(batch_size , 1920 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16)
            SwitchSequential(UNet_ResidualBlock(1920 , 640), UNet_AttentionBlock(8 , 80)),  #640 = 8*80
            #(batch_size , 960 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16)
            SwitchSequential(UNet_ResidualBlock(960 , 640), UNet_AttentionBlock(8 , 80)),   #640 = 8*80
            #(batch_size , 960 , height/16 , width/16) --> (batch_size , 640 , height/16 , width/16) --> (batch_size , 640 , height/8 , width/8)
            SwitchSequential(UNet_ResidualBlock(960 , 640), UNet_AttentionBlock(8 , 80), Upsample(640)),    #640 = 8*80
            #(batch_size , 960 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8)
            SwitchSequential(UNet_ResidualBlock(960 , 320), UNet_AttentionBlock(8 , 40)),   #320 = 8*40
            #(batch_size , 640 , height/8 , width/8) --> (batch_size , 320 , height/8 , width/8) --> (batch_size , 320 , height/4 , width/4)
            SwitchSequential(UNet_ResidualBlock(640 , 320), UNet_AttentionBlock(8 , 80)),   #320 = 8*40
            #(batch_size , 640 , height/4 , width/4) --> (batch_size , 320 , height/4 , width/4) --> (batch_size , 320 , height/4 , width/4)
            SwitchSequential(UNet_ResidualBlock(640 , 320), UNet_AttentionBlock(8 , 40)),   #320 = 8*40

        ])

    def forward(self, x , context , time):

        # x : (batch_size , 4 , height/8 , width/8)
        # context : (batch_size , seq_len/context length , dim/channels)
        # time : (1, 1280) --> might be learnable parameter of length 1280

        # saving the values for skip connections
        skip_connections = []

        # encoders
        for encoder in self.encoders:
            x = encoder(x , context , time)   #(batch_size , channels , height , width)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x , context , time)   #(batch_size , channels , height , width)

        # decoders
        for decoder in self.decoders:
            skip_connection = skip_connections.pop()
            # adding the channel dimension
            x = torch.cat((x , skip_connection), dim=1)   #(batch_size , channels , height , width)
            x = decoder(x , context , time)    #(batch_size , channels , height , width)

        return x


# this is applied at the end of unet to convert the 320 channels to 4 channels
class UNet_output_layer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.groupnorm= nn.GroupNorm(32, in_channels)  
        self.conv = nn.Conv2d(in_channels , out_channels, kernel_size=3 , padding=1)
    
    def forward(self, x):

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x

class Diffusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_embedding = TimeEmbedding(320)    # (1,320) --> (1,1280)
        self.unet = UNet()  # backbone unet model
        self.final = UNet_output_layer(320,4)   #converting 320 channels to 4 channels

    def forward(self, latent , context , time):

        #time : (1,320)
        # latent : (batch_size , 4 , height/8 , width/8)
        # context : (batch_size , seq_len/context length , dim/channels)

        time = self.time_embedding(time)
        output = self.unet(latent , context , time)  #(batch_size , 4 , height/8 , width/8) -> (batch_size , 320 , height/8 , width/8)
        output = self.final(output)  #(batch_size , 320 , height/8 , width/8) -> (batch_size , 4 , height/8 , width/8)

        return output
    

