
import torch
from torch import nn
from torch.nn import functional as F
import math





#self attention
class SelfAttention(nn.Module):

    def __init__( self , n_heads:int, d_embd: int, in_proj_bias=True , out_proj=True):
        super().__init__()

        self.in_proj = nn.Linear( d_embd, 3 * d_embd , bias= in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x:torch.Tensor , casual_mask= True):

        input_shape = x.shape
        batch_size , sequence_len , d_embd = input_shape

        intermediate_shape = (batch_size, sequence_len, self.n_heads , self.d_head)
        

        q,k,v = self.in_proj(x).chunk( 3, dim=-1)       #seperate projections

        q = q.view(intermediate_shape).transpose(1,2)
        k = k.view(intermediate_shape).transpose(1,2)
        v = v.view(intermediate_shape).transpose(1,2)

        wei = q @ k.T
        if casual_mask:     #upper triangular(values above principle diagonal is 1)--> which we have to fill with -infinity
            mask = torch.ones_like(wei , dtype=torch.bool).tril(1)
            wei.masked_fill_(mask, -torch.inf)

        wei = wei/math.sqrt(self.d_head)
        wei = F.softmax(wei, dim = -1)

        output = wei @ v        #we have transposed above to doing it back
        output = output.transpose(1,2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)      #we have not specified the parameters of the out_projections then how? 
        return output
    


    # attention block 

class Vae_AttentionBlock(nn.Module):
    
    def __init__(self , channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32 , channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):
        # x --> batchsize, features/channel , height , width

        residue = x

        n , c , h , w = x.shape

        x = x.view(n , c , h*w)    
        x = x.transpose(-1,-2)      #--> (n , c , h*w ) --> (n, h*w, c) , pixel corresponding to 1 channel are arranged along the column , so self attention is maybe learning the relationship between the pixel along different channel

        x = self.attention(x)

        x = x.transpose(-1, -2)     # to inverse the transformation

        x = x.view(n,c,h,w)

        x += residue

        return x
        
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embd , d_cross , in_proj_bias = True , out_proj_bias= True): 
        super().__init__()
        self.q_proj = nn.Linear(d_embd , d_embd, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross , d_embd, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross , d_embd, bias = in_proj_bias)

        self.out_proj = nn.Linear(d_embd, d_embd , bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd//n_heads

    def forward(self,x,y):
        # x --> latent --> batch_size , seq_len_q, dim_q
        # y --> context --> batch_size , sew_len_kv , dim_kv --> batchsize , 77, 768
        input_shape = x .shape
        batch_size , sequence_lenth , d_embd = input_shape

        intermin_shape =(batch_size , -1 ,self.n_heads , self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(intermin_shape).transpose(1,2)
        k = k.view(intermin_shape).transpose(1,2)
        v = v.view(intermin_shape).transpose(1,2)

        wei = q @ k.t(-1,-2)

        wei /= math.sqrt(self.d_head)

        wei = F.softmax(wei, dim=1)

        output = wei * v

        output= output.t(1,2).contiguous()      #what is this new function

        output = output.view(input_shape)
        output = self.out_proj(output)
   
        return output 

        

