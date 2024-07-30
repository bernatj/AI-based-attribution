# AI-based-Attribution
This repository contains the basic scripts to perform the attribution of extreme weather events based on the adaptation of the pseudo-global warming approach applied to AI-based weather models like FourCastNetv2 and Pangu Weather.

For more information on how to get to run this trainned weather models, visit the [Earth2MIP Github Repo](https://github.com/NVIDIA/earth2mip).

This repository is structured into the following subdirectories containing the relevant scripts to perform the rapid attribution of extreme events.

1. **Model init data preprocessing scripts**: 
    This directory contains all needed scripts to prepare the input files to run FourCastNet or Pangu Weather. 

    - Preprocessed CMIP6 data is needed to calculate the climate deltas applied to initial conditions. We provide the multimodel mean of 10 models interpolated to a common grid of 2.5x2.5deg. Preprocessed files needed to apply the Anthropogenic Climate change (ACC) deltas can be found in the data folder in this repository. CMIP6 historical data is available through the ESGF nodes [here](https://aims2.llnl.gov/search/cmip6/). 

2. **AI model simulation scripts**: 
    This directory contains the basic scripts to be able to run the AI-based weather models. Using the external package Earth2MIP we have tested FourCastNetv2 and Pangu Weather, but depending on your hardware you might be able to use other models too.

3. **Extreme events case studies**: 
    This folder contains jupyter notebooks used to calculate the different diagnostics for three selected case studies of extreme events. The data produced using the scripts in (2) needs to be used in order to be able to run this scripts.


## Additional remarks:
- To be able to run all these scripts please create use the enviroment file called: ```ai-attribution.yaml```

