import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, is_causal=False, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
            
        b, seq_len_q, _ = q.size()
        _, seq_len_k, _ = k.size()
        
        q = self.wq(q).view(b, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(b, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(b, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.is_causal or mask is not None:
            if self.is_causal:
                mask = torch.triu(torch.ones(seq_len_q, seq_len_k), diagonal=1).bool()
                mask = mask.to(q.device)
            if mask is not None:
                attn_score = attn_score.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
                
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        context = torch.matmul(attn_weight, v).transpose(1, 2).contiguous().view(b, seq_len_q, self.d_model)
        return self.wo(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, is_causal=False, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, mask=mask)
        x = x + self.dropout(attn_output)
        
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, is_causal=True, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, is_causal=False, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(x_norm, mask=tgt_mask)
        x = x + self.dropout(self_attn_output)
        
        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attn(x_norm, enc_output, enc_output, mask=src_mask)
        x = x + self.dropout(cross_attn_output)
        
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, num_layers=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, num_layers=6, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0/math.sqrt(2))
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embedded = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        enc_output = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, enc_output, tgt_mask, src_mask)
        
        return self.output_layer(dec_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)