#import statements
import stochastic_error_model
import numpy as np 
import pandas as pd
import math
import scipy.stats as st
from scipy.stats import norm
import statsmodels.tsa as tsa
import statsmodels.api as sm

fpath = '/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/'
ensemble = 'ensemble_1/'
fut_name = 'gfdl_ssp370'

#load in historical data
df = pd.read_csv(fpath+ensemble+'hist_annual.csv')
#load in future data
df_fut = pd.read_csv(fpath+ensemble+'fut_annual_'+fut_name+'.csv')
m = 10000

Q,D,res = stochastic_error_model.future_stoch_mvnorm(df,df_fut,m)

#make sure there are no negative runoff values - set minimum to be zero
#for both historical and future
Q[Q<0] = 0

#nyears = len(df.year.unique())
#get historical values
#Q_hist = Q[:nyears,:,:]
#get future values
#Q_fut = Q[nyears:,:,:]

np.save(fpath+ensemble+'Q',Q)