import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    scores_attn = scores

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):        
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores, _ = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output
    
    
class FeedForward(nn.Module):
    """ Positional-Wise Feed-Forward Layer """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.gelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # weight matrix, each row present one word
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.embed(x)
        return embed # * math.sqrt(self.d_model)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    

class Sampler(nn.Module):
    def __init__(self, d_model, latent_dim, variational):
        super(Sampler, self).__init__()
        self.variational = variational
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def sampling(self, mu, log_var):
        if self.variational:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.sampling(mu, log_var)
        return z, mu, log_var
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, dff, dropout):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm_1(x)
        x2 = self.attn(x, x, x, mask)
        x = x + self.dropout_1(x2)
        x = self.norm_2(x)
        x2 = self.ff(x)
        x = x + self.dropout_2(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_2 = Norm(d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_3 = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
