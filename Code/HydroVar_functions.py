#This script contains functions for generating the figures in main_figures.py for Birnbaum et al. "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability"

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

def future_stoch_mvnorm(df,df_fut,m):
    """Function that generates stochastic realizations for historical and future periods for all basins.
    Inputs:
    df: Pandas dataframe of annual runoff data in historical period.
    df_fut: Pandas dataframe of annual runoff data for future period.
    m: number of stochastic realizations.
    """
    
    basin_nms = df['basin_name'].unique() #get list of basin names
    nyears = len(df['year'].unique()) # time periods (every year)
    nbasins = len(basin_nms) # how many basins?
    nfyears = len(df_fut.year.unique()) # how many future time periods?

    # create numpy arrays for observed, deterministic model,and future deterministic projection (across all basins)
    q_det = np.reshape(df.groupby(['basin_id'],group_keys=False)['q_det'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    q_obs = np.reshape(df.groupby(['basin_id'],group_keys=False)['q_obs'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    qfut_sim = np.reshape(df_fut.groupby(['basin_id'],group_keys=False)['q_fut'].apply(lambda x:x).to_numpy(),
                       (nfyears,nbasins),order='F')

    # initialize empty storage arrays: 
    D_syn = np.zeros([nyears+nfyears,nbasins,m]) # store simulated differences
    Q_syn = np.zeros([nyears+nfyears,nbasins,m]) # store simulated runoff
    yfit_stor = np.zeros([nyears+nfyears,nbasins]) # store fitted means
    e_stor = np.zeros([nyears,nbasins]) # store errors
    sdp_stor = np.zeros([nyears+nfyears,nbasins]) # store standard deviation of prediction
    
    for j in range(nbasins): # loop through all of the basins
        boi = df[df.basin_name==basin_nms[j]] # boi = basin of interest
        boi_fut = df_fut[df_fut.basin_name==basin_nms[j]] # boi_fut = basin of interest future data
        q_obs_basin = boi.q_obs.to_numpy() # get historical reference annual runoff
        q_det_basin = boi.q_det.to_numpy()  # get historical deterministic model annual runoff
        qfut_sim_basin = boi_fut.q_fut.to_numpy() # get future projection annual runoff
        diff_basin = q_obs_basin - q_det_basin # difference between reference and deterministic model (historical)

        # fit OLS model
        X_basin = sm.add_constant(q_det_basin) #add column of ones to array of independent variable

        # calculate the mean
        beta = np.matmul(np.linalg.inv(np.matmul(X_basin.T,X_basin)),np.matmul(X_basin.T,diff_basin)) #these are the beta coefficients for OLS Y = Xb
        yfit = np.matmul(X_basin,beta) # model mean vector X*beta = yfit
        e = yfit - diff_basin # model residuals (difference between model fit and observed differences)
        sige = np.std(e) # standard deviation of model residuals

        # calculate variance of prediction
        vare = sige**2
        varyhat = vare * np.matmul(np.matmul(X_basin,np.linalg.inv(np.matmul(X_basin.T,X_basin))),X_basin.T)
        varypred = np.diag(varyhat+vare) #take diagonal elements of matrix
        
        # calculate future using model parameters from historical
        Xfut_basin = sm.add_constant(qfut_sim_basin) # add column of ones to future data
        yfit_fut = np.matmul(Xfut_basin,beta)   # Xfut*beta coefficients --> yfit future

        # #calculate variance of prediction for future
        varyhat_fut = vare*np.diag(np.matmul(np.matmul(Xfut_basin,np.linalg.inv(np.matmul(X_basin.T,X_basin))),Xfut_basin.T))
        varypred_fut = varyhat_fut+vare #take diagonal elements of matrix
        varypred_fut = np.array(varypred_fut,dtype='float')
        
        # save the values for yfit, model residuals e, and standard deviations of model residuals
        yfit_stor[:,j] = np.concatenate((yfit,yfit_fut))
        e_stor[:,j] = e
        sdp_stor[:,j] = np.concatenate((np.sqrt(varypred),np.sqrt(varypred_fut)))
    
    corr_e = np.corrcoef(e_stor.T) # calculate correlation structure across basins based on historical errors
    
    for i in range(nyears+nfyears): # now loop through time
        yfit_vals = yfit_stor[i,:]  # get mean vector for year i
        
        #build the covariance matrix for year i
        sd_vals = sdp_stor[i,:]
        sd_matrix = np.multiply.outer(sd_vals,sd_vals)
        cov_matrix = np.multiply(corr_e,sd_matrix)
            
        # generate errors from multivariate normal distribution (e*)
        D_syn[i,:,:] = np.random.multivariate_normal(yfit_vals,
                                                     cov_matrix,
                                                     m).T
        # calculate stochastic ensemble of runoff
        qdet_fut = np.concatenate((q_det,qfut_sim)) #add future deterministic to historical deterministic
        Q_syn[i,:,:] = np.tile(qdet_fut[i,:],(m,1)).T+D_syn[i,:,:] # qobs* = qdet + e*
        
    return Q_syn #return stochastic runoff ensemble

def plot_pexc(df_basin,Q_syn,df_basin_fut,Q_syn_fut,cc='#1f78b4',cc2='lightgray',plot_type='hist_det',ax=None):
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
