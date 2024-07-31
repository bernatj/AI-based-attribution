# AI-based Attribution Forecasts 

This repository contains scripts to attribute extreme weather events to Anthropogenic Climate Change (ACC) using forecasts from newly developed AI-based weather models. The methodology involves applying climate deltas (the difference between actual and preindustrial climate) of thermodynamic variables from historical simulations of global climate models (GCMs) to the initial conditions of weather forecasts. This approach is adapted from the pseudo-global warming methodology ([Brogli et al., 2023](https://doi.org/10.5194/gmd-16-907-2023)), traditionally used with regional and global atmospheric physics-based models, but here applied to AI-driven weather forecasts.

We utilize two AI-based weather models: [FourCastNetv2](https://arxiv.org/abs/2306.03838) and [Pangu-Weather](https://doi.org/10.1038/s41586-023-06185-3).

To run these models, we use the repository [https://github.com/jejjohnson/ddf.git](https://github.com/jejjohnson/ddf.git), which largely relies on the [Earth2MIP](https://github.com/NVIDIA/earth2mip) framework. This is particularly relevant if you plan to use the scripts in the `2_AI_model_simulation_scripts` directory.

This repository is organized into the following directories, each containing scripts relevant to the rapid attribution of extreme events:

1. **Model Initialization Data Preprocessing Scripts**: 
    This directory includes scripts for preparing input files for FourCastNetv2 or Pangu-Weather. 

    - Preprocessed CMIP6 data ([Eyring et al., 2016](https://doi.org/10.5194/gmd-9-1937-2016)) is required to calculate the climate deltas applied to initial conditions. We provide a multimodel mean of 10 models, interpolated to a common grid of 2.5x2.5 degrees. The necessary preprocessed files for applying the ACC deltas can be found in the DATA folder in this repository. The original CMIP6 historical data is available through the ESGF nodes [here](https://aims2.llnl.gov/search/cmip6/).

2. **AI Model Simulation Scripts**: 
    This directory contains the basic scripts needed to run the AI-based weather models. While we've tested FourCastNetv2 and Pangu-Weather using the external Earth2MIP package, other models might be compatible depending on your hardware.

3. **Extreme Events Case Studies**: 
    This folder includes Jupyter notebooks for calculating diagnostics for three selected case studies of extreme events. The data generated using the scripts in section 2 is necessary to run these notebooks.

## Installation:

### Conda (Recommended)

- clone the repository into you machine:
```bash
git clone https://github.com/bernatj/AI-based-Attribution.git
```

- create a new conda environment:
```bash
conda env create -f environments/attribution_with_ai.yaml
conda activate attribution_with_ai
```

- if you want to be able to run the models you need to manually install APEX using pip:
```bash
pip install git+https://github.com/NVIDIA/apex.git -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext"
```

- clone the repository DDP (https://github.com/jejjohnson/ddf.git) in you desired directory and modify the path to it the scripts in (2):
```bash
git clone https://github.com/jejjohnson/ddf.git
```

