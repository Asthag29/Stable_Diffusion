
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

        input_shape = x.shape   # (batch_size , sequence_len/height*width , d_embd/channels)
        batch_size , sequence_len , self.d_embd = input_shape   

        intermediate_shape = (batch_size, sequence_len, self.n_heads , self.d_head)

        q,k,v = self.in_proj(x).chunk( 3, dim=-1)       #(batch_size , sequence_len , d_embd) -> (batch_size , sequence_len , 3* d_embd) ->(chunk) -> 3 * (batch_size , sequence_len , d_embd)

        q = q.view(intermediate_shape).transpose(1,2)   # (batch_size , sequence_len , d_embd) -> (batch_size , sequence_len , n_heads , d_head) -> (batch_size , n_heads , sequence_len , d_head)
        k = k.view(intermediate_shape).transpose(1,2)   # (batch_size , sequence_len , d_embd) -> (batch_size , sequence_len , n_heads , d_head) -> (batch_size , n_heads , sequence_len , d_head)
        v = v.view(intermediate_shape).transpose(1,2)   # (batch_size , sequence_len , d_embd) -> (batch_size , sequence_len , n_heads , d_head) -> (batch_size , n_heads , sequence_len , d_head)

        wei = q @ k.T   #wei(batch_size , n_heads , sequence_len , sequence_len) = (batch_size , n_heads , sequence_len , d_head) @ (batch_size , n_heads , d_head , sequence_len) 
        if casual_mask:     # weather we want masked attention or not
            mask = torch.ones_like(wei , dtype=torch.bool).tril(1)
            wei.masked_fill_(mask, -torch.inf)

        wei = wei/math.sqrt(self.d_head)
        wei = F.softmax(wei, dim = -1)

        output = wei @ v        #output  -> (batch_size , n_heads , sequence_len , d_head)
        output = output.transpose(1,2)  # (batch_size , sequence_len , n_heads , d_head)
        output = output.reshape(input_shape)    # (batch_size , sequence_len , d_embd)
        output = self.out_proj(output)      # (batch_size , sequence_len , d_embd)

        return output

# Cross Attention
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

        # x(latent) --> batch_size , seq_len_q, dim_q
        # y(context) --> batch_size , seq_len_kv , dim_kv = (batchsize , 77, 768)
        input_shape = x .shape
        batch_size , sequence_length , d_embd = input_shape

        #dividing the embedding dim into n_heads and d_head
        intermin_shape =(batch_size , -1 ,self.n_heads , self.d_head)   #(batch_size , seq_len , n_heads , d_head)

        q = self.q_proj(x)      #(batch_size , seq_len_q , d_embd)
        k = self.k_proj(y)      #(batch_size , seq_len_kv , d_embd)
        v = self.v_proj(y)      #(batch_size , seq_len_kv , d_embd)

        q = q.view(intermin_shape).transpose(1,2)   #(batch_size , seq_len_q , d_embd) -> (batch_size , seq_len_q , n_heads , d_head) -> (batch_size , n_heads , seq_len_q , d_head)
        k = k.view(intermin_shape).transpose(1,2)   #(batch_size , seq_len_kv , d_embd) -> (batch_size , seq_len_kv , n_heads , d_head) -> (batch_size , n_heads , seq_len_kv , d_head)
        v = v.view(intermin_shape).transpose(1,2)   #(batch_size , seq_len_kv , d_embd) -> (batch_size , seq_len_kv , n_heads , d_head) -> (batch_size , n_heads , seq_len_kv , d_head)

        wei = q @ k.t(-1,-2)    #wei -> (batch_size , n_heads , seq_len_q , seq_len_kv) = (batch_size , n_heads , seq_len_q , d_head) @ (batch_size , n_heads , d_head , seq_len_kv)
        wei /= math.sqrt(self.d_head)
        wei = F.softmax(wei, dim=1)

        output = wei * v    # output -> (batch_size , n_heads , seq_len_q , d_head)
        output= output.t(1,2).contiguous()      #output(batch_size , seq_len_q , n_heads , d_head)
        output = output.view(input_shape)       #(batch_size , seq_len_q , d_embd)  , d_embd = n_heads * d_head
        output = self.out_proj(output)          # (batch_size , seq_len_q , d_embd)

        return output