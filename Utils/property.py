import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit.Chem import Descriptors
from rdkit import RDLogger, rdBase
from rdkit.Chem.rdchem import AtomValenceException
from Utils import mapper


def disable_rdkit_logging():
    """Disable RDKit whiny logging"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')
disable_rdkit_logging()


def logP(mol):
    return Descriptors.MolLogP(mol)


def tPSA(mol):
    return Descriptors.TPSA(mol)


def QED(mol):
    try:
        return Descriptors.qed(mol)
    except AtomValenceException:
        return np.nan
    

property_fn = {
    "logP": logP,
    "tPSA": tPSA,
    "QED" : QED
}


def mol_to_prop(valid_mol, property_list, n_jobs):
    prop = OrderedDict()
    for p in property_list:
        prop[p] = mapper(property_fn[p], valid_mol, n_jobs)
    return pd.DataFrame(prop)