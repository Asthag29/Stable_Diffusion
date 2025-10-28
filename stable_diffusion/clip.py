import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class ClipEmbedding(nn.Module):     #inout embedding
     
     # n_vocab : vocabulary size
     # n_embd : embedding dimension
     # n_token : sequence length / context length
    def __init__(self, n_vocab: int, n_embd:int, n_token:int):

        super().__init__()

        # converting token space to embedding dimension where each token has it's unique number which is then converted into n_embd(cause we are working in the embedding space)
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # positional embedding(all the values are initialized to zero), nn.Parameter makes it learnable(makes the gradient flow possible)
        self.position_embedding = nn.Parameter(torch.zeros(n_token,n_embd)) 

    def forward(self , tokens):

        # tokens : (batch_size , seq_len/context length)
        x = self.token_embedding(tokens)   # x : (batch_size , seq_len , n_embd) 
        x += self.position_embedding       # x : (batch_size , seq_len/context length , n_embd) + (n_token/seq_length , n_embd)  --> broadcasting happens here

        return x
    
class ClipLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)     #(batch_size , seq_len , n_embd)
        self.attention = SelfAttention(n_head, n_embd)  #(batch_size , seq_len , n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4* n_embd, n_embd)

    def forward(self , x):

        residue = x     #residue , x = (batch_size , seq_len , n_embd)
        x = self.layernorm_1(x)     #(batch_size , seq_len , n_embd)

        #masked self attention
        x = self.attention(x , casual_mask = True)  #(batch_size , seq_len , n_embd)
        x += residue    #x: (batch_size , seq_len , n_embd)

        #feedforward network after the self attention

        residue = x     #(batch_size , seq_len , n_embd)
        x = self.layernorm_2(x)     #(batch_size , seq_len , n_embd)
        x = self.linear_1(x)        #(batch_size , seq_len , n_embd) -> (batch_size , seq_len , 4*n_embd)

        # dont know why 1.702
        x = x * torch.sigmoid(1.702 * x)  

        x = self.linear_2(x)        #(batch_size , seq_len , 4*n_embd) -> (batch_size , seq_len , n_embd)
        x += residue            #x: (batch_size , seq_len , n_embd)

        return x


class Clip(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.embedding = ClipEmbedding(n_vocab=49408, n_embd=768, n_token=77)

        # applying attention with multiple head size 
        self.layers = nn.ModuleList([
            ClipLayer(n_head=12, n_embd=768) for i in range(12)
        ])
        self.layersnorms = nn.LayerNorm(768)

    def forward(self , tokens : torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long) #(batch_size , seq_len/context length)
        state = self.embedding(tokens)  # (batch_size , seq_len/context length , n_embd)

        # passing through the layers multiple times
        for layer in self.layers:
            state = layer(state)
        output = self.layersnorms(state)

        return output