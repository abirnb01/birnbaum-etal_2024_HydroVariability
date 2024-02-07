import pandas as pd
import numpy as np
import glob

def get_la_sum(df,crop_list):
    """Read in data frame of crop production and sum across irr/rfd and fertilizer amount """
    df[['crop','basin','irr_type','fert_type']] = df.landleaf.str.split('_',n=3,expand=True)
    df = df[df.crop.isin(crop_list)]    #limit to crops of interest   
    df = df.groupby(['Units','scenario','region','basin','year','irr_type'])['value'].sum().reset_index() #sum across crops and fertilizer type
    return df
    
# load in path and name of query
fpath='/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/'
queryName = 'land_alloc'

#get scenarios
queries = glob.glob(fpath+queryName+'/*.csv') #list of all queries

#let's select all the scenarios
query_select = queries[0:]

#list of crops to include
crop_list = ['CornC4', 'FiberCrop', 'FodderGrass', 'FodderHerb',
       'FruitsTree', 'Fruits', 'Legumes', 'MiscCropTree',
       'MiscCrop', 'NutsSeedsTree', 'NutsSeeds', 'OilCropTree', 'OilCrop',
       'OilPalmTree', 'OtherGrainC4', 'OtherGrain', 'Rice',
       'RootTuber', 'Soybean',
       'SugarCropC4', 'Vegetables', 'Wheat', 'FodderHerbC4', 'SugarCrop']
       
#load in each scenario
scens = pd.read_csv(query_select[0])
la = get_la_sum(scens,crop_list)
for i in range(1,len(query_select)):
    scen_new = pd.read_csv(query_select[i])
    la_new = get_la_sum(scen_new,crop_list)
    la = pd.concat([la,la_new])

la.set_index('scenario').to_csv(fpath+'combined_csvs/'+queryName+'_sum.csv')
