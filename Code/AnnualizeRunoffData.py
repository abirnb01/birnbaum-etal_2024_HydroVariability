# general useful packages
import time, os, math, random
import pandas as pd # data analysis
import numpy as np # numerical analysis

# set path to data
fpath = 'C:/Users/birnb/Documents/Tufts Research/GCIMS/birnbaum-etal_2024_HydroVariability/Data/Runoff_data/Raw_data/'
hist_obs = 'watergap2-2e_gswp3-w5e5_obsclim_histsoc_nowatermgt_qtot_basin_km3-per-mth_1901_2019.csv'
hist_mod = 'Basin_runoff_km3permonth_pm_abcd_mrtm_watergap2-2e_gswp3-w5e5_1901_2019.csv'
upcol_mod = 'gutacalib_220.npy' #recalibrated for Upper Colorado
fut_mod = 'Basin_runoff_km3permonth_gfdl-esm4_r1i1p1f1_ssp370_1850_2100.csv'

# load in "observed" data - watergap2 model that xanthos is calibrated to
# watergap2 columns are basin, year, month, and runoff value.
obs_data = pd.read_csv(fpath+hist_obs)

# load in xanthos historical data used for calibration
# xanthos columns are yearmonth, rows are basins - from 1901 to 2010
det_data = pd.read_csv(fpath+hist_mod)

# create dictionary mapping basin name to basin id
basin_dict = dict(zip(det_data.name,det_data.id))

# convert deterministic model data to same format as observed data
det_data2 = det_data.T # transpose data frame
det_data2.columns = det_data2.iloc[1] # set column names to basin names
det_data2 = det_data2.iloc[2:] # remove first two rows of dataframe (column names and associated ID, will add back later)

det_data3 = det_data2.unstack() # reformat so year,month and basin_name are columns
det_data3 = det_data3.reset_index()
det_data3 = det_data3.rename(columns={'name':'basin_name','level_1':'year_month',0:'q'}) # rename columns
det_data3['basin_id'] = det_data3['basin_name'].map(basin_dict) # create column for basin ID
det_data3['year'] = det_data3['year_month'].str[0:4].astype(int) # create column for year
det_data3['month'] = det_data3['year_month'].str[4:].astype(int) # create column for month

# merge deterministic model and observed dataframes
obs_data = obs_data.rename(columns={'basin':'basin_id'}) # rename column for basin id in observed dataframe
data = obs_data.merge(det_data3,on=['basin_id','year','month'],suffixes=['_obs','_det']) # merge dataframes
data = data.filter(['basin_name','basin_id','year','month','q_obs','q_det'])

# Convert Runoff from Monthly to Annual
data['q_det'] = data['q_det'].astype(float)
data_annual = data.groupby(['basin_id','basin_name','year'])['q_obs','q_det'].sum() #annual runoff by basin
data_annual = data_annual.reset_index()

# load in recalibrated Upper Colorado Basin
up_col = np.load(fpath+upcol_mod)
yrs = np.linspace(1901,2019,119).astype(int)
up_col_runoff = pd.DataFrame()
up_col_runoff['value'] = up_col
up_col_runoff['year'] = np.repeat(yrs,12)
up_col_runoff = up_col_runoff.groupby(['year'])['value'].sum().reset_index()

#replace the upper colorado with the recalibrated data
data_annual.loc[data_annual.basin_id==220,'q_det'] = up_col_runoff.value.to_numpy()

#read in data
fut_data = pd.read_csv(fpath+fut_mod)

#create dictionary mapping basin name to basin id
basin_dict = dict(zip(fut_data.name,fut_data.id))

#convert futulated data to same format as observed data
fut_data2 = fut_data.T #transpose data frame
fut_data2.columns = fut_data2.iloc[1] #set column names to basin names
fut_data2 = fut_data2.iloc[2:] #remove first two rows of dataframe (column names and associated ID, will add back later)

fut_data3 = fut_data2.unstack() #reformat so year,month and basin_name are columns
fut_data3 = fut_data3.reset_index()
fut_data3 = fut_data3.rename(columns={'name':'basin_name','level_1':'year_month',0:'q_fut'}) # rename columns
fut_data3['basin_id'] = fut_data3['basin_name'].map(basin_dict) # create column for basin ID
fut_data3['year'] = fut_data3['year_month'].str[0:4].astype(int) # create column for year
fut_data3['month'] = fut_data3['year_month'].str[4:].astype(int) # create column for month

fut_annual = fut_data3.groupby(['basin_id','basin_name','year'])['q_fut'].sum().reset_index() # annual runoff by basin
fut_annual = fut_annual[fut_annual.year>2019] # limit future to starting in 2020

# make sure data_annual and fut_annual are in proper order
data_annual = data_annual.sort_values(by=['basin_id','year'])
fut_annual = fut_annual.sort_values(by=['basin_id','year'])

# rename Missouri for both historical and future dataframes
data_annual.loc[data_annual.basin_name=='Missouri River Basin','basin_name'] = 'Missouri'
fut_annual.loc[fut_annual.basin_name=='Missouri River Basin','basin_name'] = 'Missouri'

#save data_annual and fut_annual
data_annual.set_index('basin_id').to_csv(fpath+'hist_annual.csv',encoding='utf-8',header=True)
fut_annual.set_index('basin_id').to_csv(fpath+'fut_annual_gfdl_ssp370.csv',encoding='utf-8',header=True)