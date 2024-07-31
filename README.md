# AI-based Attribution Forecasts 

This repository contains the basic scripts to perform the attribution of extreme weather events to Anthropogenic climate change (ACC) using forecasts of newly developed AI-based weather models. This methodology consist in taking climate deltas (difference between actual and preindustrial climate) of thermodynamical variables from historical simulations of global climate models (GCMs) and apply it to the initial conditions of the weather forecasts. This is based on the adaptation of the pseudo-global warming approach ([Brogli et. al, 2023](https://doi.org/10.5194/gmd-16-907-2023)), which has been developed to be applied to regional and global atmospheric physics-based models. Here we apply it to data-driven or AI weather forecasts.

In particular we use two AI-based weather models: [FourCastNetv2](https://arxiv.org/abs/2306.03838) and [Pangu-Weather](https://doi.org/10.1038/s41586-023-06185-3).

To be able to run these two models we use the following repository: https://github.com/jejjohnson/ddf.git which is based almost exclusively on the [Earth2MIP](https://github.com/NVIDIA/earth2mip) framework. This is specially relevant if you want to run the running script in  the `2_AI_model_simulation_scripts` directory.

This repository is structured into the following subdirectories containing the relevant scripts to perform the rapid attribution of extreme events.

1. **Model init data preprocessing scripts**: 
    This directory contains all needed scripts to prepare the input files to run FourCastNetv2 or Pangu-Weather. 

    - Preprocessed CMIP6 ([Eyring et al (2016)](https://doi.org/10.5194/gmd-9-1937-201)) data is needed to calculate the climate deltas applied to initial conditions. We provide the multimodel mean of 10 models interpolated to a common grid of 2.5x2.5deg. Preprocessed files needed to apply the ACC deltas can be found in the DATA folder in this repository. The original CMIP6 historical data is available through the ESGF nodes [here](https://aims2.llnl.gov/search/cmip6/). 
    
2. **AI model simulation scripts**: 
    This directory contains the basic scripts to be able to run the AI-based weather models. Using the external package Earth2MIP we have tested FourCastNetv2 and Pangu-Weather, but depending on your hardware you might be able to use other models too.

3. **Extreme events case studies**: 
    This folder contains jupyter notebooks used to calculate the different diagnostics for three selected case studies of extreme events. The data produced using the scripts in (2) needs to be used in order to be able to run this notebooks.


## Installation:

- Create a conda enviroment using the following file:  ```ai-attribution.yaml```
- Clone the repository DDP (https://github.com/jejjohnson/ddf.git) and modify the path in the scripts in (2).

