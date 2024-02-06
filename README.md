# birnbaum-etal_2024_HydroVariability
This repository contains the code used to generate the data and figures in the paper "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability" by Birnbaum et al., which is currently under review.

## Reproduce my results
This study consists of two primary analyses:
1. running and evaluating a stochastic watershed model for annual runoff for 235 water basins with global coverage for both historical (1901-2019) and future (2020-2100) time periods
2. running a selection of scenarios (100) through the Global Change Analysis Model (GCAM) which represents the interactions between socioeconomic-climate-land-energy-water systems.

To run the stochastic watershed model and replicate Figures 1-5, follow the instructions in Code/main_figures.ipynb (Jupyter notebook), described in more detail below. In addition, Code/stochastic_error_model.py can be used to run the stochastic model without generating any figures. The inputs to the stochastic error model (annual runoff for 235 GCAM basins) is in Data/Runoff_data. The code used to process these inputs from raw data is in Code/AnnualizeRunoffData.py.

To generate the GCAM databases used in this analysis, follow these instructions.....

To replicate the results from this analysis, follow these instructions...

## Contents
The structure of this repository is as follows:

Any questions regarding this repository should be directed to abigail.birnbaum@tufts.edu

## References
