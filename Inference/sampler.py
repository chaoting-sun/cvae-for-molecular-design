import torch
import numpy as np
import torch.nn.functional as F
from Model import get_model
from Inference.utils import top_k_logits, get_trg_mask, \
    tokenlen_gen_from_data_distribution


class Sampling:
    def __init__(self, model, kwargs):
        self.model        = model
        self.top_k        = kwargs['top_k']
        self.SRC          = kwargs['SRC']
        self.TRG          = kwargs['TRG']
        self.pad_id       = self.SRC.vocab.stoi['<pad>']
        self.sos_id       = self.TRG.vocab.stoi['<sos>']
        self.eos_id       = self.TRG.vocab.stoi['<eos>']
        self.sep_id       = self.TRG.vocab.stoi['<sep>']
        
        self.cond_dim     = kwargs['cond_dim']
        self.latent_dim   = kwargs['latent_dim']
        self.max_strlen   = kwargs['max_strlen']
        self.use_cond2dec = kwargs['use_cond2dec']
        self.decode_algo  = kwargs['decode_algo']
        self.toklen_data  = kwargs['toklen_data']
        
        self.scaler       = kwargs['scaler']
        self.device       = kwargs['device']
        self.n_jobs       = kwargs['n_jobs']
    
    def init_y(
            self,
            n,
            add_sos=True,
            sca_ids=None,
            add_sep=False
        ):
        start_ids = []
        if add_sos:
            start_ids.append(self.sos_id)
        if sca_ids is not None:
            start_ids.extend(sca_ids)
        if add_sep:
            start_ids.append(self.sep_id)
        ys = torch.from_numpy(np.stack([start_ids]*n))
        return ys

    def id_to_smi(self, ids):
        smi = ''
        for i in ids:
            if i == self.eos_id:
                break
            if i != self.sos_id:
                smi += self.TRG.vocab.itos[i]
        return smi
    
    def smi_to_id(
            self,
            smi,
            add_sos=False,
            add_sep=False,
            add_eos=False
        ):
        token = self.TRG.tokenize(smi)
        ids = []
        if add_sos:
            ids.append(self.sos_id)
        if add_sep:
            ids.append(self.sep_id)
        ids.extend([self.TRG.vocab.stoi[t] for t in token])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def sample_toklen(self, n):
        n_bin = int(self.toklen_data.max()
                  - self.toklen_data.min())
        toklens = tokenlen_gen_from_data_distribution(data=self.toklen_data, size=n, nBins=n_bin)
        toklens = toklens.reshape((-1,)) + self.cond_dim
        toklens = (np.rint(toklens)).astype(int)
        return toklens

    def tokenize_smiles(self, smiles_list, field='SRC'):
        if field == 'SRC':
            p = self.SRC.process([self.SRC.tokenize(smi)
                                  for smi in smiles_list])
            if not self.SRC.batch_first:
                return p.T
        else:
            exit('no code!')
        return p
        
    def sample_z(self, toklen, n):
        return torch.normal(
            mean=0, std=1,
            size=(n, toklen, self.latent_dim) 
        )
        
    def transform(self, prop):
        prop = self.scaler.transform(prop)
        return torch.from_numpy(prop).float()

    def decoder_input(
        self,
        ys,
        z,
        dconds=None,
        transform=False
    ):
        kwargs = {}
        kwargs['z'] = z.to(self.device)
        kwargs['trg'] = ys.to(self.device)
        if dconds is not None:
            if transform:
                dconds = self.transform(dconds)
            if not torch.is_tensor(dconds):
                dconds = torch.from_numpy(np.array(dconds))
            kwargs['dconds'] = dconds.to(self.device)
        return kwargs

    def decode(self, **kwargs):
        ys = kwargs['ys']
        break_condition = torch.zeros(ys.size(0), dtype=torch.bool)
        
        with torch.no_grad():
            for i in range(self.max_strlen - 1):
                # prepare decoder input
                kws = self.decoder_input(ys, kwargs['zs'])
                kws['src_mask'] = kwargs['src_mask']
                if self.cond_dim > 0:
                    kws['dconds'] = kwargs['dconds']
                    kws['trg_mask'] = get_trg_mask(ys, self.pad_id, self.use_cond2dec,
                                                   kwargs['dconds'])
                else:
                    kws['trg_mask'] = get_trg_mask(ys, self.pad_id, self.use_cond2dec)

                output_mol = self.model.decode(**kws)
                
                if self.use_cond2dec:
                    output_mol = output_mol[:, self.cond_dim:, :]
                
                prob = F.softmax(output_mol, dim=-1)
                prob = prob[:, -1, :]
                
                # select top k values
                if self.top_k != -1:
                    prob = top_k_logits(prob, k=self.top_k)

                # select next word
                if self.decode_algo == 'greedy':
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                
                elif self.decode_algo == 'multinomial':
                    next_word = torch.multinomial(prob, 1)
                    ys = torch.cat([ys, next_word], dim=1)
                    next_word = torch.squeeze(next_word)

                # check breaking condition
                end_condition = (next_word.to('cpu') == self.eos_id)
                break_condition = (break_condition | end_condition)
                if all(break_condition):
                    break
        return ys
    

class VaeSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def sample_smiles(self, n):
        ys = self.init_y(n, add_sos=True)

        toklen = self.sample_toklen(n)
        max_toklen = max(toklen)

        zs = self.sample_z(max_toklen, n)

        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        src_mask = torch.arange(max_toklen).expand(n,1,max_toklen) < toklen_stop_ids

        ys = ys.to(self.device)
        zs = zs.to(self.device)
        src_mask = src_mask.to(self.device)

        outs = self.decode(zs=zs, ys=ys, src_mask=src_mask)
        outs = outs.cpu().numpy()
        smiles = [self.id_to_smi(ids) for ids in outs]
        
        return smiles


class PVaeSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def sample_smiles(self, n, prop):
        ys = self.init_y(n, add_sos=True)

        prop = self.transform(prop.reshape(1,-1))
        prop = torch.tile(prop.view(1, -1), (n, 1))

        toklen = self.sample_toklen(n)        
        max_toklen = max(toklen)

        zs = self.sample_z(max_toklen, n)

        # mask

        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        src_mask = torch.arange(max_toklen).expand(n,1,max_toklen) < toklen_stop_ids

        # move to gpu

        ys = ys.to(self.device)
        zs = zs.to(self.device)
        prop = prop.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=prop, src_mask=src_mask)
        outs = outs.cpu().numpy()        
        smiles = [self.id_to_smi(ids) for ids in outs]
        
        return smiles


class ScaVaeSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def sample_smiles(self, n, scaffold):
        sca_ids = [self.TRG.vocab.stoi[e] for e
                   in self.TRG.tokenize(scaffold)]

        ys = self.init_y(n, add_sos=True, sca_ids=sca_ids, add_sep=True)

        toklen = self.sample_toklen(n)
        
        max_toklen = max(toklen)
        lat_toklen = len(sca_ids) + 1 + max_toklen

        zs = self.sample_z(lat_toklen, n)

        # mask
        
        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        toklen_stop_ids = torch.add(toklen_stop_ids, len(sca_ids)+1)
        src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids

        # move to gpu
        
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles

        outs = self.decode(zs=zs, ys=ys, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [self.id_to_smi(ids[1+len(sca_ids)+1:])
                  for ids in outs]
        return smiles
    

class PScaVaeSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)
    
    def sample_smiles(self, n, prop, scaffold):
        sca_ids = [self.TRG.vocab.stoi[e] for e
                   in self.TRG.tokenize(scaffold)]

        ys = self.init_y(n, add_sos=True, sca_ids=sca_ids, add_sep=True)

        prop = self.transform(prop.reshape(1,-1))
        prop = torch.tile(prop.view(1, -1), (n, 1))

        toklen = self.sample_toklen(n)
        
        max_toklen = max(toklen)
        lat_toklen = len(sca_ids) + 1 + max_toklen

        zs = self.sample_z(lat_toklen, n)

        # mask
        
        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        toklen_stop_ids = torch.add(toklen_stop_ids, len(sca_ids)+1)
        src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids

        # move to gpu
        
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        prop = prop.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=prop, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [self.id_to_smi(ids[1+len(sca_ids)+1:])
                  for ids in outs]
        return smiles


sampling_tools = {
    'Vae'      : VaeSampling,
    'PVae'     : PVaeSampling,
    'ScaVae'   : ScaVaeSampling,
    'PScaVae'  : PScaVaeSampling
}


def get_sampler(args, SRC, TRG, toklen_data, scaler, device):
    model = get_model(args, len(SRC.vocab), len(TRG.vocab), device)
    model = model.to(device)
    model.eval()

    kwargs = {
        'top_k'       : args.top_k,
        'latent_dim'  : args.latent_dim,
        'max_strlen'  : args.max_strlen,
        'use_cond2dec': args.use_cond2dec,
        'decode_algo' : args.decode_algo,
        'n_jobs'      : args.n_jobs,
        'toklen_data' : toklen_data,
        'cond_dim'    : len(args.property_list),
        'scaler'      : scaler,
        'device'      : device,
        'SRC'         : SRC,
        'TRG'         : TRG,
    }
    
    return sampling_tools[args.model_type](model, kwargs)