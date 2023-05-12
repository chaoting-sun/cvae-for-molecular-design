import torch
import torch.nn as nn
from Model.layers import (
    EncoderLayer,
    DecoderLayer,
    Embeddings,
    PositionalEncoding,
    Norm,
    get_clones,
    Sampler
)


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, 
                 latent_dim, nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.nconds = nconds
        self.variational = variational
        self.embed_sentence = Embeddings(d_model, vocab_size)
        
        self.norm = Norm(d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(h, d_model, dff, dropout), N)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)
        
        if nconds > 0:
            self.embed_cond2enc = nn.Linear(nconds, d_model*nconds)

    def forward(self, src, mask, dconds):
        x = self.embed_sentence(src)
        
        if self.nconds > 0:
            assert dconds is not None
            cond2enc = self.embed_cond2enc(dconds)
            cond2enc = cond2enc.view(dconds.size(0), dconds.size(1), -1)
            x = torch.cat([cond2enc, x], dim=1)
        
        x = self.pe(x)
        
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Pass N decoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim,
                 nconds, dropout, use_cond2dec, use_cond2lat):
        super(Decoder, self).__init__()
        self.N = N
        self.nconds = nconds
        self.d_model = d_model
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        self.embed = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(h, d_model, dff, dropout), N)
        self.norm = Norm(d_model)
        
        if use_cond2dec and nconds > 0:
            self.embed_cond2dec = nn.Linear(nconds, d_model*nconds)
        if use_cond2lat and nconds > 0:
            self.embed_cond2lat = nn.Linear(nconds, d_model*nconds)

    def forward(self, trg, e_outputs, src_mask, trg_mask, dconds):
        x = self.embed(trg)
        e_outputs = self.fc_z(e_outputs)

        if self.use_cond2dec and self.nconds > 0:
            cond2dec = self.embed_cond2dec(dconds)
            cond2dec = cond2dec.view(dconds.size(0), dconds.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
            
        elif self.use_cond2lat and self.nconds > 0:
            cond2lat = self.embed_cond2lat(dconds)
            cond2lat = cond2lat.view(dconds.size(0), dconds.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)

        x = self.pe(x)

        if self.use_cond2lat and self.nconds > 0:
            cond_mask = torch.ones_like(torch.unsqueeze(dconds, -2),
                                        dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Vae(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048,
                 h=8, latent_dim=64,  dropout=0.1, nconds=3,
                 use_cond2dec=False, use_cond2lat=False, variational=True):
        super(Vae, self).__init__()
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        self.sampler = Sampler(d_model, latent_dim, variational)
        self.out = nn.Linear(d_model, trg_vocab)
        
        if use_cond2dec and nconds > 0:
            self.prop_fc = nn.Linear(trg_vocab, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask, econds=None):
        x = self.encoder(src, src_mask, econds)
        z, mu, log_var = self.sampler(x)
        return z, mu, log_var

    def decode(self, trg, z, src_mask, trg_mask, dconds=None):
        x = self.decoder(trg, z, src_mask, trg_mask, dconds)
        return self.out(x)

    def forward(self, src, trg, src_mask, trg_mask,
                econds=None, dconds=None):
        z, mu, log_var = self.encode(src, src_mask, econds)
        output = self.decode(trg, z, src_mask, trg_mask, dconds)

        if self.use_cond2dec:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        else:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol, mu, log_var, z
