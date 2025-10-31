from torch import nn
from torch.nn import functional as F
from attention import SelfAttention , CrossAttention

#Attention Block
class UNet_AttentionBlock(nn.Module):

    def __init__(self,n_heads , n_embd , d_context = 768):

        super().__init__()
        
        channels = n_embd * n_heads

        self.groupnorm = nn.GroupNorm(32 , channels , eps = 1e-6)
        self.conv_input = nn.Conv2d(channels , channels, kernel_size=1 , padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads=n_heads , d_embd=channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads=n_heads, d_embd=channels , d_cross=d_context , in_proj_bias = False)

        self.layernorm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1= nn.Linear(channels, 4 * channels * 2)  #later divided into 2 don't know why
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1 , padding=0)
    
    def forward(self, x , context):

        # x(data) = batch_size , features/channels , height , width
        # context(clip_model) = batch_size , seq_len/context_len , dim/channels

        residue_long = x
        
        x = self.groupnorm(x)   #(batch_size , channels , height , width)
        x = self.conv_input(x)  #(batch_size , channels , height , width)

        n , c , h , w = x.shape  #(batch_size , channels , height , width)

        x= x.view(n , c , h*w)  #(batch_size , channels , height*width)
        x = x.transpose(-1,-2)      #(batch_size , height*width , channels)

        # normalization + self_attention 

        residue_short = x   #(batch_size , height*width , channels)
        x = self.layernorm_1(x)     #(batch_size , height*width , channels)
        x = self.attention_1(x)   #(batch_size , height*width , channels)
        x += residue_short      #(batch_size , height*width , channels)

        # normalization + cross_attention
        residue_short = x       #(batch_size , height*width , channels)
        x = self.layernorm_2(x)     #(batch_size , height*width , channels)
        x = self.attention_2(x , context)       #(batch_size , height*width , channels)
        x += residue_short      #(batch_size , height*width , channels)

        residue_short = x    #(batch_size , height*width , channels)

        # normalization + feed forward with geglu and skip connection
        x = self.layernorm_3(x)    #(batch_size , height*width , channels)

        x , gate = self.linear_geglu_1(x).chunk(2 , dim=-1)     #(batch_size , height*width , channels) -> (batch_size , height*width , 2*4*channels) -> 2 * (batch_size , height*width , 4*channels)
        x = x * F.gelu(gate)    #(batch_size , height*width , 4*channels)
        x = self.linear_geglu_2(x)    #(batch_size , height*width , 4*channels) -> (batch_size , height*width , channels)

        x += residue_short  #(batch_size , height*width , channels)

        x = x.transpose(-1,-2)    #(batch_size , channels , height*width)
        x = x.view(n , c , h , w)   #(batch_size , channels , height , width)
        x = self.conv_output(x)    #(batch_size , channels , height , width)
        
        x += residue_long   #(batch_size , channels , height , width)

        return x


# Residual Block

class UNet_ResidualBlock(nn.Module):

    def __init__(self, in_channels , out_channels, n_time=1280):

        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3 , padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1 , padding=0)

    def forward(self, feature , time):

        # feature : (batch_size , in_channels , height , width)
        # time : (1, 1280) --> might be learnable parameter of length 1280
        residue = feature       #(batch_size , in_channels , height , width)

        feature = self.groupnorm_feature(feature)   #(batch_size , in_channel , height , width)
        feature = F.silu(feature)                 #(batch_size , in_channel , height , width)
        feature = self.conv_feature(feature)    #(batch_size , out_channel , height , width)

        time = F.silu(time)     #(1, 1280)
        time = self.linear_time(time)       #(1,1280) -> (1, out_channels)

        # same time value is added to all the pixels of a particular channel
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)  #(batch_size , out_channels , height , width)
        merged = self.groupnorm_merged(merged)      #(batch_size , out_channels , height , width)
        merged = F.silu(merged)                     #(batch_size , out_channels , height , width)
        merged = self.conv_merged(merged)           #(batch_size , out_channels , height , width)

        return merged + self.residual_layer(residue)


