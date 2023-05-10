from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def get_mol(smi_or_mol):
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smi_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smi_or_mol


def get_canonical(smi_or_mol):
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smi_or_mol)
    elif isinstance(smi_or_mol, Chem.Mol):
        mol = smi_or_mol
    else:
        return None

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold(smi_or_mol):
    mol = get_mol(smi_or_mol)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol)
