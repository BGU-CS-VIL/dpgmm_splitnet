# Supplementary Code for the Paper: 
## Common Failure Modes of Subcluster-based Sampling in Dirichlet Process Gaussian Mixture Models - and a Deep-learning Solution

This repository contains the code to train the SplitNet models and run in within the DPMMSubCluster.jl package.


* An example notebook with some results (`Examples.ipynb`) is available.
* To install the run the DPMMSubCluster.jl package, refer to it's documentation.
* To train your own SplitNet models, read the `Readme` file in the splitnet directory.
* To run Sub-* methods with your own trained models, please update path to the trained models in the config file at `DPMMSubClusters.jl/configs/basic.yaml`