import os
import torch
from collections import OrderedDict
from Model import Vae, Cvae


model_class = {
    'Vae'    : Vae,
    'PVae'   : Cvae,
    'ScaVae' : Cvae,
    'PScaVae': Cvae,
}


model_path = {
    'Vae'    : os.path.join('.', 'Data', 'vae.pt'),
    'PVae'   : os.path.join('.', 'Data', 'pvae.pt'),
    'ScaVae' : os.path.join('.', 'Data', 'scavae.pt'),
    'PScaVae': os.path.join('.', 'Data', 'pscavae.pt'),
}


def _load_state(model, model_path, device):
    # map_location = { 'cuda:%d' % 0: 'cuda:%d' % rank }
    model_state = torch.load(model_path, device)['model_state_dict']    
    if list(model_state.keys())[0].split('.')[0] == 'module':
        model_state = OrderedDict([(k[7:], v) for k, v in model_state.items()])
    model.load_state_dict(model_state)
    return model 


def get_model(args, src_vocab_len, trg_vocab_len, device):
    hyper_params = {
        'src_vocab'   : src_vocab_len,
        'trg_vocab'   : trg_vocab_len,
        'N'           : args.N,
        'd_model'     : args.d_model,
        'dff'         : args.d_ff,
        'h'           : args.H,
        'latent_dim'  : args.latent_dim,
        'dropout'     : args.dropout,
        'use_cond2dec': args.use_cond2dec,
        'use_cond2lat': args.use_cond2lat,
        'nconds'      : len(args.property_list)
    }
    model = model_class[args.model_type](**hyper_params)
    return _load_state(model, model_path[args.model_type], device)