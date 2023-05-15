import os
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict

from Inference import get_sampler
from Utils import get_logger, mapper, get_mol, \
    murcko_scaffold, get_canonical, mol_to_prop, \
    get_ssf, run_moses_benchmark, get_property_error


def is_number(a):
    try:
        float(a)
        return True
    except ValueError:
        return False


def input_settings(property_list, use_scaffold):
    kwargs = {}

    # number of generated SMILES

    while True:
        n = input('How many molecules do you want to produce?\n\n')
        if n.isdigit() and int(n) > 0:
            n = int(n)
            break
        else:
            print('Please enter a natural number greater than 0!')

    # scaffold

    if use_scaffold:
        while True:
            _scaffold = input(f'\nInput target scaffold:\n\n'
                              f'Ex1: N=c1[nH]c2ccccc2[nH]1\n'
                              f'Ex2: c1ccc(OCn2ccnc2)cc1\n'
                              f'Ex3: O=C1CN=C(c2ccccc2)c2ccccc2N1\n\n')
            scaffold = murcko_scaffold(_scaffold)
            if scaffold is not None:
                break
            else:
                print('Invalid SMILES:', _scaffold)
        kwargs['scaffold'] = scaffold
    
    # target properties

    if property_list:
        while True:
            _trg_prop = input(f'\nInput target properties:\n'
                              f'Suggested properties: '
                              f'logP: 0.03 - 4.97   '
                              f'tPSA: 17.92 - 112.83   '
                              f'QED: 0.58 - 0.95\n'
                              f'Ex: 3.25 58.6 0.85\n\n'
                             )
            p = _trg_prop.split()
            if len(p) != 3 or not is_number(p[0]) or not is_number(p[1]) or not is_number(p[2]):
                print('Wrong format:', _trg_prop)
            else:
                trg_prop = np.array([float(e) for e in p])
                break
        kwargs['prop'] = trg_prop

    return n, kwargs


def add_args(parser):
    # changeable parameters
    
    parser.add_argument('-model_type', type=str, required=True,
                        help="Vae, PVae, ScaVae, or PScaVae")
    parser.add_argument('-use_scaffold', action='store_true',
                        help="conditioned by a scaffold or not")
    parser.add_argument('-property_list', nargs='+', default=[],
                        help='conditioned by properties or not')
    parser.add_argument('-decode_algo', type=str, default='multinomial',
                        help='multinomial or greedy')
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('-n_jobs', type=int, default=1,
                        help='number of cpus')    
    parser.add_argument('-top_k', type=int, default=-1, help="""Controls the
                       number
                        of top-k predictions to consider during decoding.
                        Only the tokens with the highest probabilities
                        are considered. Set to -1 to consider all tokens""")

    # fixed parameters

    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-all_property_list', default=['logP', 'tPSA', 'QED'])

    # model architecture

    parser.add_argument('-N', type=int, default=6)
    parser.add_argument('-H', type=int, default=8)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-variational', type=bool, default=True)
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    logger = get_logger()
    LOG = logger(name='sampling', log_path='./record.log')
    LOG.info(args)

    # get gpu
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cuda':
            print('No available gpu! Use cpu!')
    else:        
        device = 'cpu'

    LOG.info('get device: %s', device)

    # get distributions of token length

    toklen_data = pd.read_csv(os.path.join('Data', 'toklen_list.csv'))

    # get smiles field

    field_suffix = "_sep" if args.use_scaffold else ""
    SRC = pickle.load(open(os.path.join('Data', f'SRC{field_suffix}.pkl'), 'rb'))
    TRG = pickle.load(open(os.path.join('Data', f'TRG{field_suffix}.pkl'), 'rb'))

    LOG.info('Get scaler')

    scaler = joblib.load(os.path.join('Data', f'scaler.pkl'))

    save_folder = './Gen'
    os.makedirs(save_folder, exist_ok=True)

    LOG.info(f'Create a new folder if not exists: {save_folder}')

    if args.property_list and args.use_scaffold:
        prefix = 'psca'
    elif args.property_list and not args.use_scaffold:
        prefix = 'p'
    elif not args.property_list and args.use_scaffold:
        prefix = 'sca'
    elif not args.property_list and not args.use_scaffold:
        prefix = 'uc'
    
    smiles_path = os.path.join(save_folder, f"{prefix}_smiles.csv")
    prop_path = os.path.join(save_folder, f"{prefix}_prop.csv")
    metric_path = os.path.join(save_folder, f"{prefix}_metric.csv")

    LOG.info('Get a sampler')

    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

    LOG.info('Get settings')

    n, kwargs = input_settings(args.property_list, args.use_scaffold)

    LOG.info('Generate SMILES')

    i = 0
    batch_size = 256
    if os.path.exists(smiles_path):
        os.remove(smiles_path)

    while n > 0:
        LOG.info(f'{n} SMILES left...')

        smiles = sampler.sample_smiles(n=min(n, batch_size), **kwargs)
        n -= batch_size

        _smiles = pd.DataFrame(smiles, columns=['smiles'])
        if i == 0:
            _smiles.to_csv(smiles_path, index=False)
        else:
            _smiles.to_csv(smiles_path, mode='a', index=False, header=False)
        i += 1

    LOG.info('Get MOSES metrics and molecular properties...')
    
    samples = pd.read_csv(smiles_path)

    metrics = run_moses_benchmark(samples=samples['smiles'], n_jobs=args.n_jobs)
    metrics = OrderedDict([(k, v) for k, v in metrics.items()])

    samples = samples.dropna(subset=['smiles'])
    mols = mapper(get_mol, samples['smiles'], args.n_jobs)
    valid_mol = [m for m in mols if m is not None]
    valid_smi = mapper(get_canonical, valid_mol, args.n_jobs)

    gen_prop = mol_to_prop(valid_mol, args.all_property_list, args.n_jobs)

    gen_prop.insert(0, 'smiles', valid_smi)
    gen_prop.to_csv(prop_path, index=False)

    if args.property_list:
        error = get_property_error(kwargs['prop'], gen_prop,
                                   args.property_list)
        metrics.update(error)

    if args.use_scaffold:
        ssf = get_ssf(valid_mol, kwargs['scaffold'], args.n_jobs)
        metrics['SSF'] = ssf

    metrics = OrderedDict([(k, [v]) for k, v in metrics.items()])
    metrics = pd.DataFrame(metrics)

    metrics.to_csv(metric_path, index=False)

    LOG.info('Results:\n%s', metrics)