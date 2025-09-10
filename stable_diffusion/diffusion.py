import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention , CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd , 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd , n_embd)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x
class Upsample(nn.Module):

    def __call__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels , kernel_size=3, padding=1)

    def forward(self,x):
        x = F.interpolate(x , scale_factor=2 , mode="nearest")      #differnce between interpolate an dupsample
        return self.conv(x)

class UNet_AttentionBlock(nn.Module):
    def __init__(self,n_heads , n_embd , d_context = 786):
        super().__init__()
        channels = n_embd * n_heads

        self.groupnorm = nn.GroupNorm(32 , channels , eps = 1e-6)
        self.conv_input = nn.Conv2d(channels , channels, kernel_size=1 , padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads , channels, in_proj_bias=False)

        self.layernorm_2 = CrossAttention(n_heads, channels , d_context , in_proj_bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1= nn.Linear(channels)
        self.linear_geglu_2 = nn.Linear(channels, 4 * channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1 , padding=0)
    
    def forward(self, x , context):
        # x = batch_size , features , height , width
        # context = natch , seq_len , dim 

        residue_long = x
        
        x = self.groupnorm(x)

        x = self.conv_input(x)

        n , c , h , w = x.shape

        x= x.view(n , c , h*w)

        x = x.transpose(-1,-2)

        # normalization + self_attention 

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short

        residue_short = x

        # normalization + skip_connection
        x = self.layernorm_1(x)
        
        # cross_attention
        self.attention_2(x , context)
        x += residue_short

        residue_short = x

        # normalization + ff with geglu and skip connection 
        x = self.layernorm_3(x)

        x , gate = self.linear_geglu_1(x).chunk(2 , dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        x = x.tranpose(-1,-2)
        x = x.view(n , c , h , w)

        return self.conv_output(x) + residue_long


class UNet_ResidualBlock(nn.Module):
    def __init__(self,in_channel , out_channel, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channel)
        self.conv_feature = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.linera_time = nn.Linear(n_time, out_channel)

        self.groupnorm_merged = nn.GroupNorm(32, out_channel)
        self.conv_merged = nn.Conv2d(out_channel, out_channel, kernel_size=3 , padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else :
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1 , padding=0)

    def forward(self, feature , time):

        # feature battch_size , inchannel , height , width
        #time (1, 1280)
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linera_time(time)

        merged = feature +time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    
class SwitchSequential(nn.Sequential):

    def forward(self, x , context , time):
        for layer in self:
            if isinstance(layer , UNet_AttentionBlock):
                x = layer(x , context)
            elif isinstance(layer , UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoders = nn.Module([
            SwitchSequential(nn.Conv2d(4 , 320 , kernel_size=3 , padding=1)),
            SwitchSequential(UNet_ResidualBlock(320 , 320), UNet_AttentionBlock(8,40)),
            SwitchSequential(UNet_ResidualBlock(320 , 320), UNet_AttentionBlock(8,40)),

            SwitchSequential(nn.Conv2d(320, 320 , kernel_size=3, stride=2 , padding=1)),
            SwitchSequential(UNet_ResidualBlock(320 , 640), UNet_AttentionBlock(8,40)),
            SwitchSequential(UNet_ResidualBlock(640 , 640), UNet_AttentionBlock(8,40)),
            
            SwitchSequential(nn.Conv2d(640, 640 , kernel_size=3, stride=2 , padding=1)),
            SwitchSequential(UNet_ResidualBlock( 640, 1280), UNet_AttentionBlock(8,160)),
            SwitchSequential(UNet_ResidualBlock(128 , 1280), UNet_AttentionBlock(8,160)),

            SwitchSequential(nn.Conv2d(128, 1280 , kernel_size=3, stride=2 , padding=1)),
            SwitchSequential(UNet_ResidualBlock(1280 , 1280)),
            SwitchSequential(UNet_ResidualBlock(1280 , 1280)),

        ])
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280,1280),
            
            UNet_AttentionBlock(8,160),

            UNet_ResidualBlock(1280,1280),
        )

        self.decoders = nn.Module([
             #differnce betweeen module and modulelist
            SwitchSequential(UNet_ResidualBlock(2560 , 1280)),
            SwitchSequential(UNet_ResidualBlock(2560 , 1280)),

            
            SwitchSequential(UNet_ResidualBlock(2560 , 1280), Upsample(1280)),
            SwitchSequential(UNet_ResidualBlock(2560 , 1280), UNet_AttentionBlock(8,160)),
            
           
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8,160)),
            SwitchSequential(UNet_ResidualBlock(1920 , 1280), UNet_AttentionBlock(8,160), Upsample(1280)),

           
            SwitchSequential(UNet_ResidualBlock(1920 , 640), UNet_AttentionBlock(8 , 80)),
            SwitchSequential(UNet_ResidualBlock(960 , 640), UNet_AttentionBlock(8 , 80)),
            SwitchSequential(UNet_ResidualBlock(960 , 640), UNet_AttentionBlock(8 , 80), Upsample(640)),
            
            SwitchSequential(UNet_ResidualBlock(960 , 320), UNet_AttentionBlock(8 , 40)),

            SwitchSequential(UNet_ResidualBlock(640 , 320), UNet_AttentionBlock(8 , 80)),

            SwitchSequential(UNet_ResidualBlock(640 , 320), UNet_AttentionBlock(8 , 40)),




        ])
class UNet_output_layer(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.groupnorm= nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels , out_channels, kernel_size=3 , padding=1)
    
    def forward(self, x):

        x = self.groupnorm(x)

        x = F.silu(x)

        x= self.conv(x)

        return x

class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_output_layer(320,4)

    def forward(self, latent , context , time):
        #time :m(1,320)

        #(1,320 )--> (1, 1280)
        time = self.time_embedding(time)
        output = self.unet(latent , context , time)

        output = self.final(output)

        return output
    

