import pandas as pd
import numpy as np
import geopandas as gpd

#load in CSV for water withdrawals (separated by sufrace water or groundwater)
fpath = '/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/combined_csvs/'
spath = '/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/pickled_data/'
queryName = 'water_withdrawals_source'
ww = pd.read_csv(fpath+queryName+'.csv')  #read CSV
#split up surface water and groundwater
ww_gw = ww[ww.subresource.str.contains('groundwater')] #just groundwater
ww_sw = ww[ww.subresource.str.contains('runoff')] #just surface water

#filter to remove column for region (just care about basin) and to sum up different groundwater grades
ww = ww.groupby(['scenario','Units','resource','year'])['value'].sum().reset_index()
ww_gw = ww_gw.groupby(['scenario','Units','resource','year'])['value'].sum().reset_index()
ww_sw = ww_sw.groupby(['scenario','Units','resource','year'])['value'].sum().reset_index()
#these lines of code are used to clean up the basin name column (originally "resource") in the 3 dataframes
ww[['basin','ww']] = ww['resource'].str.split('_water withdrawals',n=2,expand=True)
ww = ww.drop(['ww','resource'],axis=1)
ww_gw[['basin','ww']] = ww_gw['resource'].str.split('_water withdrawals',n=2,expand=True)
ww_gw = ww_gw.drop(['ww','resource'],axis=1)
ww_sw[['basin','ww']] = ww_sw['resource'].str.split('_water withdrawals',n=2,expand=True)
ww_sw = ww_sw.drop(['ww','resource'],axis=1)
#save the cleaned up data as pickles:
ww.to_pickle(spath+'ww')
ww_gw.to_pickle(spath+'ww_gw')
ww_sw.to_pickle(spath+'ww_sw')

#load in CSV for water withdrawals (separated by sufrace water or groundwater)
queryName = 'water_withdrawals_irrig'
ww_irrig = pd.read_csv(fpath+queryName+'.csv')  #read CSV
#filter to remove column for region (just care about basin)
ww_irrig = ww_irrig.groupby(['scenario','Units','subsector','year'])['value'].sum().reset_index()
#these lines of code are used to clean up the basin name column (originally "resource") in the 3 dataframes
ww_irrig[['crop','basin']] = ww_irrig['subsector'].str.split('_',n=2,expand=True)
#save the cleaned up data as pickles:
ww_irrig.to_pickle(spath+'ww_irrig')

#load in maximum available runoff query output
queryName = 'max_subresource'
mr = pd.read_csv(fpath+queryName+'.csv') 
mr = mr[mr.subresource=='runoff'] #limit to just runoff (as opposed to all resources from the query)
mr = mr.groupby(['scenario','Units','resource','year'])['value'].sum().reset_index()
#clean up name to show basin instead of "resource"
mr[['basin','ww']] = mr['resource'].str.split('_water withdrawals',n=2,expand=True)
mr = mr.drop(['resource','ww'],axis=1)
mr.to_pickle(spath+'mr')

#water price
queryName = 'water_price'
wp = pd.read_csv(fpath+queryName+'.csv') 
#split up the name of the basin (take out _water withdrawals), make two columns basin and ww (ww is empty)
wp[['basin','ww']] = wp['market'].str.split('_water withdrawals',n=2,expand=True)
mktnames = wp.basin.unique() #list of market names
bsnnames = [] #empty list to store basin names
for mkt in mktnames: #loop through the markets
    n = len(mkt) #length of string for market name
    bsnnames.append(mkt[:int(n/2)]) #the basin name is half the length of the market name
#make a dictionary for market names to basin names
mkt_to_bsn = dict(zip(mktnames, bsnnames))
#replace basin names to not be duplicated using the dictionary
wp['basin'] = wp['basin'].map(mkt_to_bsn) #map the correct basin names
#drop extra column
wp = wp.drop(['ww'],axis=1)
#save as pickle
wp.to_pickle(spath+'wp')

#save others as pickles too
queryName = 'ag_production_allbasin_sum'
agprod_sum = pd.read_csv(fpath+queryName+'.csv')
agprod_sum.to_pickle(spath+'agprod_sum')
queryName = 'land_alloc_sum'
landalloc_sum = pd.read_csv(fpath+queryName+'.csv')
landalloc_sum.to_pickle(spath+'landalloc_sum')
queryName = 'land_alloc_indus'
landalloc_indus = pd.read_csv(fpath+queryName+'.csv')
landalloc_indus.to_pickle(spath+'landalloc_indus')
