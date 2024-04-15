# %%
import torch
import torch.nn as nn
import math

# %%
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #converting all the tokens in the vocabulary to vector of d_model dimension
        
    def forward(self, x):
        return self.embedding(x)/math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        # Create an encoding corresponding to every position in the sequence
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(dropout) 
        #  pe --> positional embedding    
        pe = torch.zeros(seq_len, d_model) #create a matrix of the size with zeroes
        
        # This generates a 2D tensor shape of which is seq_len x 1 -- elements are 0 to 511
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) 
            
        #dim = d_model/2 (positional encoding)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        #sin and cos (operation on all rows, i.e., every position in the sequence)
        #div_term decides what is the modification  in positions within the embedding of a token 
        #position refers to the position of the token in the sequence
        pe[:,0::2] = torch.sin(position/div_term) #every even term in embedding for every position in the sequence
        pe[:,1::2] = torch.cos(position/div_term) #every odd term in embedding for every position in the sequence
        
        #adding a batch dimension in the begininning 
        #(seq_len x d_model) -> (1 x seq_len x d_model) 
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) #not updated during training
        
    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert self.d_model % h == 0, "d_model/h should be an integer"
        
        self.d_k = d_model//h #every head gets a dimension of d_k
        self.w_q = nn.Linear(d_model, d_model, bias = False) #Wq
        self.w_k = nn.Linear(d_model, d_model, bias = False) #Wk
        self.w_v = nn.Linear(d_model, d_model, bias = False) #Wv
        self.w_o = nn.Linear(d_model, d_model, bias = False) #Wo (for concatenating and then producing d_model output)
        self.dropout = nn.Dropout(dropout)    
        
    @staticmethod #for reusabilty, no need of the instance, it can be called directly
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] #we have separated it along the embedding dimension
        
        #batch x h x seq_len x d_k -> batc x h x seq_len x seq_len 
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None:
            #whereever maks is required, replace the value with -inf so that it becomes zero while taking softmax
            attention_scores.masked_fill_(mask == 0, -1e-9)
        
        attention_scores = attention_scores.softmax(dim=-1) #along the seq_length dimension
        
        return (attention_scores @ value), attention_scores #return the final matrix and attention scores
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) #batch x seq_len x d_model -> batch x seq_len x d_model
        key = self.w_k(k)
        value = self.w_v(v)
        
        #batch x seq_len x d_model -> batch x seq_len x h x d_k -> btach x h x seq_len x d_k (through transposing)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #concatenating across the embedding dimension
        #batch x h x seq_len x d_k -> (batch, seq_len, h, d_k) -> --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x) #(batch, seq_len, d_model) x (batch x d_model x d_model) -> (batch, seq_len, d_model)
        
 #LayerNormalization is defined as a subclass of nn.Module, making it a custom module in the PyTorch framework. 
 #This allows it to integrate seamlessly with other PyTorch layers and functionalities.       
class LayerNormalization(nn.Module):
    #This is the constructor method for the LayerNormalization class. It initializes a new instance of this class.
    #features: an integer that specifies the number of features (or the dimensionality) of the input tensors that will be normalized. 
    #This is often the size of the embeddings or hidden layers.
    #eps: a small float added to the standard deviation to prevent division by zero. It defaults to 10**-6.
    def __init__(self, features: int, eps: float=10**-6) -> None:
        super().__init__() #features refers to the embedding/hidden size dimension
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #learnable # Multiplied
        self.bias = nn.Parameter(torch.zeros(features)) #learnable # Added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) #batch x seq_len x 1 (across of every token embdedding)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias
    

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, h: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forwad(self, x, sublayer): #sublayer -> previous layer
        #in paper, it's multihead attention first and then the layer normalization
        #in this implementation, it is layer normalization and then multihead attention
        return x + self.dropout(sublayer(self.norm(x))) 
    
     
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: #d_ff -> dimension of feedforward layer
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
     

#single encoder block     
class EncoderBlock(nn.Module):
    
    """residual_connections are not included as arguments in the __init__ method 
    because they are a standard, integral part of the class's architecture."""
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask): #padding mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x 
    
    
class Encoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x) 
    

class DecoderBlock(nn.Module):
    
    def __init__(self, features:int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()   
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block  
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) #3 residual connections
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)  
        return x


class Decoder(nn.Module):
    
    def __init__(self, features: int, layers:int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask): #different input and masks compared to encoder
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
    
#projecting the embedding to vocabulary       
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) 
        
    #batch x seq_len x d_model -> batch x seq_len x vocab_size            
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)
    
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    #defining multiole forward methods for encoder, decoder and attention
    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512, 
                      N: int = 6, #number of encoders and decoders
                      h: int = 8, #heads
                      dropout: float = 0.1, 
                      d_ff: int = 2048):    
                        
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) #since sequence length is same for source and target, it can be same as well
    
    encoder_blocks = []
    for i in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks = []
    for i in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    #initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer