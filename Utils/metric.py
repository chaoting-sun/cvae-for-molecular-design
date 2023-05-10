import moses
from collections import OrderedDict
from Utils import mapper, murcko_scaffold, get_canonical


def run_moses_benchmark(samples, n_jobs=1):
    ptest = moses.dataset.get_statistics('test')
    ptest_scaffolds = moses.dataset.get_statistics('test_scaffolds')

    metrics = moses.get_all_metrics(
        gen=samples,
        n_jobs=n_jobs,
        test=moses.get_dataset('test'),
        ptest=ptest,
        test_scaffolds=moses.get_dataset('test_scaffolds'),
        ptest_scaffolds=ptest_scaffolds
    )
    return metrics


def get_ssf(valid_mol, trg_scaffold, n_jobs=1):
    # same scaffold fraction
    trg_scaffold = get_canonical(trg_scaffold)
    sca = mapper(murcko_scaffold, valid_mol, n_jobs)
    sca = [e for e in sca if isinstance(e, str)]
    return sum([1 for s in sca if s == trg_scaffold]) / len(valid_mol)


def get_property_error(trg_prop, gen_prop, property_list):
    errors = OrderedDict()

    for j, p in enumerate(property_list):
        print(gen_prop[p])
        print(trg_prop[j])

        delp = gen_prop[p] - trg_prop[j]
        mse = delp.mean()
        mae = delp.abs().mean()
        sd = delp.std()
        
        errors[f'{p}-MSE'] = mse
        errors[f'{p}-MAE'] = mae
        errors[f'{p}-SD'] = sd
    return errors
