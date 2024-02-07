import pandas as pd
import numpy as np
import glob

def get_prod_sum(df,crop_list):
    """Read in data frame of crop production and sum across irr/rfd and fertilizer amount """
    df = df[df.sector.isin(crop_list)]
    df[['crop','basin']] = df['subsector'].str.split('_',n=2,expand=True)
    df = df.groupby(['Units','scenario','region','basin','sector','year'])['value'].sum().reset_index() 
    return df
    
# load in path and name of query
fpath='/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/'
queryName = 'ag_production_allbasin'

#get scenarios
queries = glob.glob(fpath+queryName+'/*.csv') #list of all queries

#let's select all the scenarios
query_select = queries[0:]

#list of crops to include
crop_list = ['Corn', 'FiberCrop', 'FodderGrass', 'FodderHerb',
       'Fruits', 'Legumes', 'MiscCrop', 'NutsSeeds', 'OilCrop', 'OilPalm',
       'OtherGrain', 'Rice', 'RootTuber', 'Soybean',
       'SugarCrop', 'Vegetables', 'Wheat']  #not biomass, Pasture, Forest
       
#load in each scenario
scens = pd.read_csv(query_select[0])
agprod = get_prod_sum(scens,crop_list)
for i in range(1,len(query_select)):
    scen_new = pd.read_csv(query_select[i])
    agprod_new = get_prod_sum(scen_new,crop_list)
    agprod = pd.concat([agprod,agprod_new])

agprod.set_index('scenario').to_csv(fpath+'combined_csvs/'+queryName+'_sum.csv')
