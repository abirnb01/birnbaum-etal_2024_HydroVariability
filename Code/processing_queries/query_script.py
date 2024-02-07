import pandas as pd
import numpy as np
import glob

def combine_scenarios(fpath,queryName):
    #get scenarios
    queries = glob.glob(fpath+queryName+'/*.csv') #list of all queries

    #let's select all the scenarios
    query_select = queries[0:]

    #load in all scenarios and combine
    scens = pd.read_csv(query_select[0])
    for i in range(1,len(query_select)):
        scen_new = pd.read_csv(query_select[i])
        scens = pd.concat([scens,scen_new])

    scens.set_index('scenario').to_csv('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/combined_csvs/'+queryName+'.csv')

fpath='/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/'
queryName = 'primary_energy'
combine_scenarios(fpath,queryName)