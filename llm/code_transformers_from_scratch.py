import torch
import torch.nn as nn
import math

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        bs, seq_len, d_model = x.size()

        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Split into multiple heads
        Q = Q.view(bs, self.n_heads, seq_len, self.d_k)
        K = K.view(bs, self.n_heads, seq_len, self.d_k)
        V = V.view(bs, self.n_heads, seq_len, self.d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)

        # Concatenate heads
        output = output.view(bs, seq_len, d_model)
        output = self.out(output)
        return output

# Cross-Attention Mechanism
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, encoder_outputs):
        bs, seq_len, d_model = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(encoder_outputs)
        V = self.v_linear(encoder_outputs)

        Q = Q.view(bs, self.n_heads, seq_len, self.d_k)
        K = K.view(bs, self.n_heads, encoder_outputs.size(1), self.d_k)
        V = V.view(bs, self.n_heads, encoder_outputs.size(1), self.d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)

        output = output.view(bs, seq_len, d_model)
        output = self.out(output)
        return output

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model, n_heads)
        self.cross_attention = CrossAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs):
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attention(x, encoder_outputs)
        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

# Encoder Stack
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Decoder Stack
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x, encoder_outputs):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, encoder_outputs)
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, vocab_size)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, vocab_size)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        encoder_outputs = self.encoder(src)
        decoder_outputs = self.decoder(tgt, encoder_outputs)
        return self.fc_out(decoder_outputs)

# Example usage
d_model = 512
n_heads = 8
d_ff = 2048
n_layers = 6
vocab_size = 10000

model = Transformer(d_model, n_heads, d_ff, n_layers, vocab_size)
src = torch.randint(0, vocab_size, (32, 20))  # (batch_size, src_seq_len)
tgt = torch.randint(0, vocab_size, (32, 20))  # (batch_size, tgt_seq_len)

output = model(src, tgt)
print(output.shape)  # (32, 20, vocab_size)