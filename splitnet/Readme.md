# Supplementary Code for the Paper: 
## Common Failure Modes of Subcluster-based Sampling in Dirichlet Process Gaussian Mixture Models - and a Deep-learning Solution

This repository contains the code to train the SplitNet models.


# Create a conda environment with the requirements file:

```Bash
conda create --name <env_name> python=3.8 --file requirements.txt
```

Make sure to activate that environment.

# How to run the code:


The following training configuration are available:

```Bash
# for 2D model:
configs/set_transformer_2d.yaml

# For 3D model:
configs/set_transformer_2d.yaml

# For 10D model:
configs/set_transformer_10d.yaml

# For 20D model:
configs/set_transformer_20d.yaml


```


### To run training with a specific configuration, run:


```Bash
python train.py --config-name=set_transformer_2d.yaml
```

### To view the various dataset used for training and testing, see the Jupyter Notebook at:


```Bash
notebooks/synthetic_datasets.ipynb
```
