import torch
import numpy as np
from torch.autograd import Variable
from pathos.multiprocessing import ProcessingPool as Pool


def mapper(fn, obj, n_jobs):
    if n_jobs == 1:
        res = list(map(fn, obj))
    else:
        with Pool(n_jobs) as pool:
            res = pool.map(fn, obj)
    return res


def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF >= a)-1


def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)


def tokenlen_gen_from_data_distribution(data, size, nBins):
    count_c, bins_c = np.histogram(data, bins=nBins)

    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    return tokenlen_list


def nopeak_mask(trg_size, use_cond2dec, pad_idx, cond_dim=0):
    np_mask = np.triu(np.ones((1, trg_size, trg_size)), k=1).astype('uint8')
    if use_cond2dec == True:
        cond_mask = np.zeros((1, cond_dim, cond_dim))
        cond_mask_upperright = np.ones((1, cond_dim, trg_size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, trg_size, cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask*pad_idx


def get_cond_mask(conditions):
    cond_mask = torch.unsqueeze(conditions, -2)
    return torch.ones_like(cond_mask, dtype=bool)


def get_trg_mask(target, pad_id, use_cond2dec, conditions=None):
    trg_mask = (target != pad_id).unsqueeze(-2)
    if use_cond2dec:
        cond_mask = get_cond_mask(conditions)
        trg_mask = torch.cat([cond_mask, trg_mask], dim=2)
    cond_dim = 0 if conditions is None else conditions.size(-1)
    np_mask = nopeak_mask(target.size(1), use_cond2dec, pad_id, cond_dim)
    np_mask = np_mask.to(target.device)
    
    return trg_mask & np_mask


def top_k_logits(prob, k=4):
    v, ix = torch.topk(prob, k=k)
    out = prob.clone()
    out[out < v[:, [-1]]] = 1E-6
    return out