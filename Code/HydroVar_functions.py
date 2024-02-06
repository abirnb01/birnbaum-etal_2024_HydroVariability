#This script contains functions for generating the figures in main_figures.py for Birnbaum et al. "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_NSE(obs,sim):
    """ Returns Nash-Sutcliffe Efficiency
    Input:
    obs - array of observed runoff record
    sim - array of simulated runoff record """
    return 1 - (np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2))

def calc_MSE(obs,sim):
    """ Returns Mean Squared Error
    Input:
    obs - array of observed runoff record
    sim - array of simulated runoff record """
    n = len(obs)
    return (1/n)*np.sum((obs-sim)**2)

def plot_pexc(df_basin,Q_syn,df_basin_fut,Q_syn_fut,cc='#1f78b4',cc2='#a6cee3',plot_type='hist_det',ax=None):
    """Plot exceedance probability for observed, deterministic model, and min/max of stochastic realizations,
    future determinsitic model, and min/max of future stochastic ensemble (or some combination of these datasets).
    Modified from Ghazal Shabestnaipour's Flow Exceedance function developed for 
    Shabestanipour et al. (2023) in Water Resources Research, found:
    https://github.com/gshabestani/LRM-Squannacook/tree/SWM/Func_Lib_SWM.py
    
    Inputs:
    df_basin: annual runoff data for historical period for a single GCAM basin. Columns are basin_name, basin_id,
    q_obs (observed runoff), q_det (simulated runoff), year.
    Q_syn: numpy array containing stochastic results.
    cc: color to plot determinitic model. Default is from paired color palette.
    ax: axis to plot on. """
    
    bsn = df_basin['basin_name'].unique()[0]
    nyears = len(df_basin.year.unique())
    
    Q_stoch = pd.DataFrame(Q_syn)
    m = Q_syn.shape[1]
    
    nfyears = Q_syn_fut.shape[0]
    Q_stoch_fut = pd.DataFrame(Q_syn_fut)
    
    # historical
    plot_df=pd.DataFrame(np.zeros([nyears,m])) # create empty data frame with mxn columns (to store stochastic realizations), #rows = # points in time series
    plot_df['q_obs_sort'] = df_basin.sort_values(by='q_obs',ascending=False)['q_obs'].to_numpy() # sorted observations, largest to smallest
    plot_df['q_det_sort'] = df_basin.sort_values(by='q_det',ascending=False)['q_det'].to_numpy() # sorted deterministic model, largest to smallest
    plot_df['rank'] = plot_df.reset_index().index + 1 # create column for rank of data (ordered largest to smallest)
    plot_df['p_exc'] = plot_df['rank']/(len(plot_df)+1) # calculate exceedance as rank/length of dataset + 1

    # future
    plot_df_fut=pd.DataFrame(np.zeros([nfyears,m])) #create empty data frame with mxn columns (to store stochastic realizations), #rows = # points in time series
    plot_df_fut['q_fut_sort'] = df_basin_fut.sort_values(by='q_fut',ascending=False)['q_fut'].to_numpy() #sorted observations, largest to smallest
    plot_df_fut['rank'] = plot_df_fut.reset_index().index + 1 #create column for rank of data (ordered largest to smallest)
    plot_df_fut['p_exc'] = plot_df_fut['rank']/(len(plot_df_fut)+1) #calculate exceedance as rank/length of dataset + 1
    plot_df_fut['p_nonexc'] = 1-plot_df_fut['p_exc']
   
    # populate first mxn columns of simulated model dataframe with sorted stochastic realizations, largest to smallest
    for i in range(m):
        plot_df[i]= Q_stoch.sort_values(by=i,ascending=False).iloc[:,i].to_numpy() # historical
        plot_df_fut[i]= Q_stoch_fut.sort_values(by=i,ascending=False).iloc[:,i].to_numpy() # future
    
    # simplify data to plot
    S = plot_df.iloc[:,:m].to_numpy() # get the stochasitc realizations (first 100 columns of Simulation)
    S_max = np.max(S,axis=1) # across the realizations, what is the maximum largest value?
    S_min = np.min(S,axis=1) # across the realizations, what is the minimum smallest value?

    Sfut = plot_df_fut.iloc[:,:m].to_numpy() #get the stochasitc realizations (first 100 columns of Simulation)
    Sfut_max = np.max(Sfut,axis=1) # across the realizations, what is the maximum largest value?
    Sfut_min = np.min(Sfut,axis=1) #across the realizations, what is the minimum smallest value?

    # plot
    #fig,ax = plt.subplots(1,1)
    ax = ax
    ax.plot(norm.ppf(plot_df['p_exc']),plot_df['q_obs_sort'], color='black',lw=2,label='Historical Reference')
    
    if plot_type == 'hist_det' or plot_type== 'hist_stoch' or plot_type== 'all':
        ax.plot(norm.ppf(plot_df['p_exc']),plot_df['q_det_sort'], color=cc,lw=2,label='Historical Deterministic Model')
    if plot_type == 'hist_stoch' or plot_type== 'all':
        ax.fill_between(norm.ppf(plot_df['p_exc']),S_max,S_min,color=cc2,alpha=0.5,label='Historical Stochastic Ensemble')
    if plot_type == 'fut_det' or plot_type=='fut_stoch' or plot_type== 'all':
        ax.plot(norm.ppf(plot_df_fut['p_exc']),plot_df_fut['q_fut_sort'], color='#e31a1c',lw=2,label='Future Deterministic Projection')
    if plot_type == 'fut_stoch' or plot_type=='all':
        ax.fill_between(norm.ppf(plot_df_fut['p_exc']),Sfut_max,Sfut_min,color='#fb9a99',alpha=0.3,label='Future Stochastic Ensemble')
  
    #label plot
    ax.set_title(bsn)
    ax.set_ylabel('Annual Runoff ($km^3$)')
    ax.set_xlabel('Exceedance Probability')
    ax.set_xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999])
    ax.set_xlim(-3,3)
    ax.grid()
    
    return ax