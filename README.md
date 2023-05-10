# CVAE-for-Molecular-Design

## About the Project

Welcome to this GitHub repository, where we present a molecular design method based on a Conditional Variational Autoencoder Transformer (CVAETF). This model is tailored for property and structural constraints and has been trained on a dataset of approximately 1,580,000 molecules from the MOSES benchmarking platform. Our approach incorporates property constraints such as logP (partition coefficient), tPSA (topological polar surface area), and QED (quantitative estimate of drug-likeness), while also considering structural constraints via the Murcko scaffold. The properties and scaffold computations are performed using RDKit.

## Getting Started

1. Clone the repository
```bash
git clone git@github.com:chaoting-sun/cvae-for-molecular-design.git
```

3. Install the dependencies
```bash
conda cvae create -f environment.yml
```

4. 

https://drive.google.com/file/d/1foabzpRW2j3-JwyufG39Q7bq8NeSb2ed/view?usp=share_link


1. Please provide the command followed by the required properties and scaffold for generating SMILES.


```bash
python -u sample.py -use_scaffold -property_list logP tPSA QED -model_type PScaVae
```


