# birnbaum-etal_2024_HydroVariability
This repository contains the code used to generate the data and figures in the paper "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability" by Birnbaum et al., which is currently submitted to a journal for review.

## Reproduce my results
This study consists of two primary analyses:
1. running and evaluating a stochastic watershed model for annual runoff for 235 water basins with global coverage for both historical (1901-2019) and future (2020-2100) time periods
2. running a selection of 100 future runoff scenarios through the Global Change Analysis Model (GCAM) which represents the interactions between socioeconomic-climate-land-energy-water systems and exploring the multisector impacts of future hydrologic variability.

To run the stochastic watershed model and replicate Figures 1-5, follow the instructions in Code/main_figures.ipynb (Jupyter notebook), described in more detail below. In addition, Code/stochastic_error_model.py can be used to run the stochastic model without generating any figures for a user-specified number of stochastic realizations. The inputs to the stochastic error model (annual runoff for 235 GCAM basins for historical and future time periods) is in Data/Runoff_data. The code used to process these inputs from raw data (Data/Runoff_data/Raw_data) is in Code/AnnualizeRunoffData.py.

To select the scenarios to run through GCAM, we used the Code/stochastic_error_model.py for 10,000 realizations and selected 100 scenarios evenly across the distribution of cumulative runoff 2070-2100 in the Indus basin. The code used to select the scenarios and process the output to the appropriate CSV format for GCAM is available in the Jupyter notebook Code/supplement_figures.ipynb. The CSV files are available in Data/Runoff_data/SWM_csv.

To run the 100 runoff scenarios through GCAM, we first convert the runoff from CSV format to XML using Code/csv_to_xml.R. The resulting XML files are available in Data/Runoff_data/SWM_xml. We then generate a unique GCAM v6.0 configuration file for each scenario. The code used to generate these configuration files is in Code/create_config_files.ipynb and the resulting configuration files are in Data/GCAM_config_files. To install and run GCAM scenarios, follow the instructions on https://github.com/JGCRI/gcam-core/releases/tag/gcam-v6.0.

We use the Code/query_request.R script to query relavent GCAM outputs which are stored in pickle or CSV format in Data/GCAM_queries in the corresponding Zenodo data repository. The code used to produce Figures 6-8 is in Code/main_figures.ipynb. Code used to produce all supplement figures is in Code/supplement_figures.ipynb.

## Contents
The structure of this repository is as follows:
1. Code
   *  AnnualizeRunoffData.py: Python script with code used to process raw input data (monthly runoff at basin-scale for historical and future time periods) into annual runoff
   *  HydroVar_functions.py: Python script with functions used in main_figurs.ipynb and supplement_figures.ipynb, including calculating NSE and spatial correlation matrix
   *  csv_to_xml.R: R script used to convert stochastic watershed model output from CSV to XML format to run in GCAM
   *  main_figures.ipynb: Jupyter Notebook with Python code used to produce figures in main text of manuscript
   *  stochastic_error_model.py: Python script used to run stochastic watershed model for user-specified number of realizations. Used in main_figures.ipynb and supplement_figures.ipynb
   *  supplement_figures.ipynb: Jupyter Notebook with Python code used to produce supplemental figures in manuscript
   *  create_config_files.ipynb: Jupyter Notebook with Python code used to create the 100 GCAM configuration files
     
2. Data:
   * Runoff_data: folder containing annual runoff at basin scale
     - fut_annual_gfdl_ssp370.csv: CSV file of future projection from Xanthos (climate forcing GFDL-ESM4 with SSP370 scenario) 2020-2100, used in main_figures.ipynb
     - hist_annual.csv: CSV file of historical reference (WaterGap2) and Xanthos 1901-2019, used in main_figures.ipynb
     - Raw_data: folder containing raw monthly runoff at basin-scale
       - WaterGAP2
       - Xanthos
       - future
       - Guta
     - SWM_csv: folder containing CSVs of stochastic watershed model output. There are 100 CSVs, one for each scenario
     - SWM_xml: folder containing XMLs of stochastic watershed model output (GCAM input in XML format). There are 100 XMLs, one for each scenario
   *  Shapefiles_for_mapping: folder containing shapefiles and data used to make map figures
     - basin_to_country_mapping.csv: CSV file used to map GCAM basins to countries, used in supplement_figures.ipynb to create appropriate CSV format of runoff output for conversion to XML
     - basin_to_region_mapping.csv: CSV file used to map GCAM basins to regions, used in supplement_figures.ipynb to create appropriate CSV format of runoff output for conversion to XML
     - gcam_basins.shp: shapefile of 235 GCAM basins
     - gcam_regions.shp: shapefile of 32 GCAM regions
   *  GCAM_config_files: folder containing 100 GCAM configuration files, one for each runoff scenario run through GCAM

3. README.md: this document, which contains instructions for reproducing the results of this study and the contents of the data repository.

Any questions regarding this repository should be directed to abigail.birnbaum@tufts.edu
