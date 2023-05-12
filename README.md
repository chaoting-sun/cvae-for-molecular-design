# CVAE-for-Molecular-Design

## About the Project

Welcome to this GitHub repository, where we present a molecular design method based on a Conditional Variational Autoencoder Transformer (CVAETF). This model is tailored for property and structural constraints and has been trained on a dataset of approximately 1,580,000 molecules from the MOSES benchmarking platform. Our approach incorporates property constraints such as logP (partition coefficient), tPSA (topological polar surface area), and QED (quantitative estimate of drug-likeness), while also considering structural constraints via the Murcko scaffold. The properties and scaffold computations are performed using RDKit.

## Getting Started
(1) Clone the repository in your command prompt:
```bash
git clone git@github.com:chaoting-sun/cvae-for-molecular-design.git
```

(2) Create an environment
```bash
cd cvae-for-molecular-design
conda env create -n cvae -f ./env.yml # create a new environment named cvae
conda activate cvae # activate the environment
```

(3) Download the model from [google cloud](https://drive.google.com/drive/folders/1CpF0UfM_SQlYFeABO42vyAfMJW89dTco) and move to ./Data
- vae.pt: unconditioned model
- pvae.pt: property conditioned model
- scavae.pt: scaffold conditioned model
- pscavae.pt: property and scaffold conditioned model

(4) Run the "sample.py" script with the desired options. Use the following commands:
```bash
# unconditioned sampling
python -u ./sample.py -model_type Vae

# property conditioned sampling
python ./sample.py -property_list logP tPSA QED -model_type PVae

# scaffold conditioned sampling
python ./sample.py -use_scaffold -model_type ScaVae

# property and scaffold conditioned sampling
python ./sample.py -use_scaffold -property_list logP tPSA QED -model_type PScaVae
```
Note: If you have a GPU, you can add the -use_gpu flag to the above commands to utilize it.

(5) The results will be saved in the "./Gen" directory:
- *_smiles.csv: contains all of the generated SMILES.
- *_prop.csv: contains all valid SMILES and their properties (logP, tPSA, QED).
- *_metrc.csv: contains some metrics that evaluate the generated SMILES.

## Reference
- model structure - modified from [Hyunseung-Kim/molGCT](https://github.com/Hyunseung-Kim/molGCT)
- property computation - [rdkit/rdkit](https://github.com/rdkit/rdkit)
- evaluation metrics
    - most metrics: [molecularsets/moses](https://github.com/molecularsets/moses)
    - SSF (same scaffold fraction): [devalab/molgpt](https://github.com/devalab/molgpt)