# birnbaum-etal_2024_HydroVariability
This repository contains the code used to generate the data and figures in the paper "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability" by Birnbaum et al., which is currently in revision.

## Reproduce our results
This study consists of two primary analyses:
1. running and evaluating a stochastic watershed model for annual runoff for 235 water basins with global coverage for both historical (1901-2019) and future (2020-2100) time periods
2. running a selection of 100 future runoff scenarios through the Global Change Analysis Model (GCAM) which represents the interactions between socioeconomic-climate-land-energy-water systems and exploring the multisector impacts of future hydrologic variability.

To run the stochastic watershed model and replicate Figures 1-5, follow the instructions in Code/main_figures.ipynb. In addition, Code/stochastic_error_model.py can be used to run the stochastic model without generating any figures for a user-specified number of stochastic realizations. The inputs to the stochastic error model (annual runoff for 235 GCAM basins for historical and future time periods) is in Data/Runoff_data. The code used to process these inputs from raw data (Data/Runoff_data/Raw_data) is in Code/AnnualizeRunoffData.py. As mentioned in the manuscript, deterministic model values were updated for the Upper Colorado and Iceland basins. Code doing this updating procedure is in update_colorado_iceland_basins.ipynb.

To select the scenarios to run through GCAM, we used the Code/stochastic_error_model.py for 10,000 realizations and selected 100 scenarios evenly across the distribution of cumulative runoff 2070-2100 in the Indus basin. The code used to select the scenarios and process the output to the appropriate CSV format for GCAM is available in the Jupyter notebooks Code/pick_scenarios.ipynb and Code/supplement_figures.ipynb. CSV output is available in Data/GCAM_config_files/stochastic_runoff_csv/.

To run the 100 runoff scenarios through GCAM, we first convert the runoff from CSV format to XML using Code/csv_to_xml.R. The resulting XML files are in Data/GCAM_config_files/stochastic_runoff_xml/. We then generate a unique GCAM v6.0 configuration file for each scenario. The code used to generate these configuration files is in Code/create_config_files.ipynb and an example configuration file is in Data/GCAM_config_files/configuration_stochastic_runoff_1.xml. The default GCAM configuration file with no changes is available at Data/GCAM_config_files/configuration_ref.xml. To install and run GCAM scenarios, follow the instructions on https://github.com/JGCRI/gcam-core/releases/tag/gcam-v6.0.

We use the Code/processing_queries/query_request.R script to query relavent GCAM outputs which are stored in pickle or CSV format in Data/GCAM_queries in the corresponding Zenodo data repository (too large to store on GitHub). The code used to produce Figures 6-7 is in Code/main_figures.ipynb. Code used to produce supplement figures is in Code/supplement_figures.ipynb (except for the code for Figure S5 which is in Code/main_figures.ipynb).

## Contents
The structure of this repository is as follows:
1. Code
   *  AnnualizeRunoffData.py: Python script with code used to process raw input data (monthly runoff at basin-scale for historical and future time periods) into annual runoff
   *  create_config_files.ipynb: Jupyter Notebook with Python code used to create the 100 GCAM configuration files
   *  csv_to_xml.R: R script used to convert stochastic watershed model output from CSV to XML format to run in GCAM
   *  hydrovar_environment.yml: yml file that contains all the packages/appropriate versions for Python needed to run main_figures.ipynb and supplement_figures.ipynb
   *  HydroVar_functions.py: Python script with functions used in main_figurs.ipynb and supplement_figures.ipynb, including calculating NSE and spatial correlation matrix
   *  main_figures.ipynb: Jupyter Notebook with Python code used to produce figures in main text of manuscript
   *  pick_scenarios.ipynb: Jupyter Notebook with Python code used to pick 100 scenarios out of 10,000 ensemble members to run through GCAM
   *  stochastic_error_model.py: Python script used to run stochastic watershed model for user-specified number of realizations. Used in main_figures.ipynb and supplement_figures.ipynb
   *  supplement_figures.ipynb: Jupyter Notebook with Python code used to produce supplemental figures in manuscript
   *  update_colorado_iceland_basins.ipynb: Jupyter Notebook with Python code used to update historical deterministic model for Upper Colorado and Iceland basins
   *  processing_queries: folder containing code/scripts relavent to processing GCAM queries
     - cleanup_queries.py: Python script containing code that was used to pickle queries or "clean up" results into readable format
     - landalloc_allbasin_script.py: Python script used to combine CSVs/process land allocation query for all scenarios
     - query_request.R: R script used to query GCAM database outputs from all scenarios and save in CSV format, originally run on HPC cluster.
     - query_script.py: Python script to CSVs of query output for all scenarios
     - query_xml: folder containing XMLs of individual GCAM queries
         - land_alloc.xml: query for detailed land allocation for all GCAM basins
         - land_alloc_indus.xml: query for detailed land allocation for Indus basin
         - max_subresource.xml: query used for getting maximum annual runoff for all GCAM basins
         - water_price.xml: query used for extracting shadow price of water for all GCAM basins
         - water_withdrawals_irrig.xml: query for water withdrawals for irrigation
         - water_withdrawals_source.xml: query for water withdrawals split by runoff versus groundwater
     
2. Data:
   * Runoff_data: folder containing annual runoff at basin scale
     - fut_annual_gfdl_ssp370.csv: CSV file of future projection from Xanthos (climate forcing GFDL-ESM4 with SSP370 scenario) 2020-2100, used in main_figures.ipynb
     - hist_annual.csv: CSV file of historical reference (WaterGap2) and Xanthos 1901-2019, used in main_figures.ipynb
     - Raw_data: folder containing raw monthly runoff at basin-scale
       - Basin_runoff_km3permonth_gfdl-esm4_r1i1p1f1_ssp370_1850_2100.csv: Xanthos output for future period, monthly runoff at GCAM basin scale from Zhao et al (2023). Climate forcing data is from CMIP6 GFDL-ESM4 from Krasting et al. (2018)
       - Basin_runoff_km3permonth_pm_abcd_mrtm_watergap2-2e_gswp3-w5e5_1901_2019.csv: Xanthos output for historical period, monthly runoff at GCAM basin scale
       - Basin_guta-runoff_km3peryear_watergap2-2e_gswp3-w5e5_historical_1901_2019: Xanthos output from Guta Abeshu used to recalibrate Upper Colorado and Iceland basins
       - watergap2-2e_gswp3-w5e5_obsclim_histsoc_nowatermgt_qtot_basin_km3-per-mth_1901_2019.csv: WaterGap2 output for historical period, monthly runoff at GCAM basin scale, processed version of data that is originally from Gosling et al (2023).
   *  Shapefiles_for_mapping: folder containing shapefiles and data used to make map figures
     - basin_to_country_mapping.csv: CSV file used to map GCAM basins to countries, used in supplement_figures.ipynb to create appropriate CSV format of runoff output for conversion to XML
     - basin_to_region_mapping.csv: CSV file used to map GCAM basins to regions, used in supplement_figures.ipynb to create appropriate CSV format of runoff output for conversion to XML
     - gcam_basins.shp: shapefile of 235 GCAM basins
     - gcam_regions.shp: shapefile of 32 GCAM regions
   *  GCAM_files: folder containing example GCAM configuration file, csv of stochastic runoff, xml of stochastic runoff, and GCAM reference scenario configuration file.
   *  GCAM_queries: folder containing processed query output from 100 GCAM scenarios. Available on corresponding Zenodo data repository.

3. README.md: this document, which contains instructions for reproducing the results of this study and the contents of the data repository.

Any questions regarding this repository should be directed to abigail.birnbaum@tufts.edu

## References
Gosling, S. N., Müller Schmied, H., Burek, P., Grillakis, M., Guillaumot, L., Hanasaki, N., … Schewe, J. (2023). ISIMIP3a Simulation Data from the Global Water Sector (Version 1.0) [Data set]. ISIMIP Repository. http://doi.org/10.48364/ISIMIP.398165

Krasting, J. P., John, J. G., Blanton, C., McHugh, C., Nikonov, S., Radhakrishnan, A., … Zhao, M. (2018). NOAA-GFDL GFDL-ESM4 model output prepared for CMIP6 CMIP (Version 20230220) [Data set]. Earth System Grid Federation. http://doi.org/10.22033/ESGF/CMIP6.1407

Zhao, M., Wild, T., & Vernon, C. (2023). Xanthos Output Dataset Under ISIMIP3b Selected CMIP6 Scenarios: 1850 - 2100 [Data set]. MultiSector Dynamics - Living, Intuitive, Value-adding, Environment. http://doi.org/10.57931/2280839
