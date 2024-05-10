#This script contains functions for generating the figures in main_figures.py for Birnbaum et al. "Characterizing the Multisectoral Impacts of Future Global Hydrologic Variability"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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
    ax.plot(norm.ppf(plot_df['p_exc']),plot_df['q_obs_sort'], color='black',lw=2,label='Reference')
    
    if plot_type == 'hist_det' or plot_type== 'hist_stoch' or plot_type== 'all':
        ax.plot(norm.ppf(plot_df['p_exc']),plot_df['q_det_sort'], color=cc,lw=2,label='Deterministic Model')
    if plot_type == 'hist_stoch' or plot_type== 'all':
        ax.fill_between(norm.ppf(plot_df['p_exc']),S_max,S_min,color=cc2,alpha=0.5,label='Historical Stochastic Ensemble')
    if plot_type == 'fut_det' or plot_type=='fut_stoch' or plot_type== 'all':
        ax.plot(norm.ppf(plot_df_fut['p_exc']),plot_df_fut['q_fut_sort'], color='#e31a1c',lw=2,label='Deterministic Projection')
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
 
def model_test(df,rs,m):
    """This function is used to test performance of SWM developed in this work using a 70/30 split of 
    training/testing data.
    
    Inputs:
    df: dataframe of annual runoff with columns basin_id, basin_name, year, q_obs, q_det
    rs: random seed
    m: # stochastic realizations to generate
    
    Outputs:
    Q_reorder: stochastic runoff output for m realizations, in time order
    ind_list: list of indices that are training and testing
    ntrain: # training years
    """
    
    basin_nms = df['basin_name'].unique() #get names of basins
    nyears = len(df['year'].unique()) #get number of years
    nbasins = len(basin_nms) #get number of basins

    #create numpy arrays for observed and deterministic model of historical annual runoff
    q_obs = np.reshape(df.groupby(['basin_id'],group_keys=False)['q_obs'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    q_det = np.reshape(df.groupby(['basin_id'],group_keys=False)['q_det'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    q_diff = q_obs - q_det #difference between obs and det (what we're modeling)

    indices = np.arange(nyears) #get numpy array of indices
    
    #split the data into training and test
    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(q_det, q_diff, indices,test_size=0.3, random_state=rs)
    ntrain = len(X_train) #length of training dataset
    
    #create empty storage arrays
    D_syn = np.zeros([nyears,nbasins,m])
    Q_syn = np.zeros([nyears,nbasins,m])
    yfit_stor = np.zeros([nyears,nbasins])
    e_stor = np.zeros([ntrain,nbasins])
    sdp_stor = np.zeros([nyears,nbasins])
    
    #create ind_list variable that indicates the indices of the training and testing data (so we can put back in order)
    ind_list = np.concatenate((ind_train,ind_test))
    #create ind_reorder array that will be used to reorder back to original time series
    ind_reorder = np.zeros(nyears)
    for i in range(nyears):
        ind_reorder[i] = np.where(ind_list==i)[0]
    ind_reorder = ind_reorder.astype(int)
    
    for j in range(nbasins):#loop through all of the basins
        q_basin_train = X_train[:,j] #select basin j training data
        q_basin_test = X_test[:,j] #select basin j testing data
        diff_basin_train = y_train[:,j] #select basin j differences for training data
        diff_basin_test = y_test[:,j] #select basin j differences for testing data
        
        X_basin_train = sm.add_constant(q_basin_train) #add column of ones to array of independent variable

        #calculate mean and variance of prediction
        beta = np.matmul(np.linalg.inv(np.matmul(X_basin_train.T,X_basin_train)),
                         np.matmul(X_basin_train.T,diff_basin_train)) #these are the beta coefficients for OLS Y = Xb
        yfit_train = np.matmul(X_basin_train,beta)
        e = yfit_train - diff_basin_train
        sige = np.std(e) #standard deviation of fitted model minus D
        vare = sige**2
        varyhat_train = vare * np.matmul(np.matmul(X_basin_train,
                                                   np.linalg.inv(np.matmul(X_basin_train.T,X_basin_train))),X_basin_train.T)
        varypred_train = np.diag(varyhat_train+vare) #take diagonal elements of matrix

        #Now apply model fit to training data onto the testing data!
        X_basin_test = sm.add_constant(q_basin_test)
        yfit_test = np.matmul(X_basin_test,beta)   #Xfut*beta coefficients --> yfit future
        varyhat_test = vare*np.diag(np.matmul(np.matmul(X_basin_test,
                                                        np.linalg.inv(np.matmul(X_basin_train.T,X_basin_train))),X_basin_test.T))
        varypred_test = varyhat_test+vare #take diagonal elements of matrix
        varypred_test = np.array(varypred_test,dtype='float')

        # save the values for yfit, model residuals e, and standard deviations of model residuals
        yfit_stor[:,j] = np.concatenate((yfit_train,yfit_test))
        e_stor[:,j] = e
        sdp_stor[:,j] = np.concatenate((np.sqrt(varypred_train),np.sqrt(varypred_test)))
        
    #create matrix of correlation in the errors
    corr_e = np.corrcoef(e_stor.T) #size of e is # training years x #basins
    
    #reorder q_det to match ind_list order
    q_det_reorder = q_det[ind_list]
    
    #loop through years to sample stochastically from errors
    for i in range(nyears):
        yfit_vals = yfit_stor[i,:] #get mean vector for year i
        #build the covariance matrix for year i
        sd_vals = sdp_stor[i,:]
        sd_matrix = np.multiply.outer(sd_vals,sd_vals)
        corr_e = np.corrcoef(e_stor.T)
        cov_matrix = np.multiply(corr_e,sd_matrix)

        #generate errors from multivariate normal distribution
        D_syn[i,:,:] = np.random.multivariate_normal(yfit_vals,cov_matrix,m).T
        Q_syn[i,:,:] = np.tile(q_det_reorder[i,:],(m,1)).T+D_syn[i,:,:]#qobs* = qsyn = qdet + D*
    
    #reorder Q_syn to the correct order based on ind_reorder
    Q_reorder = Q_syn[ind_reorder]
        
    return Q_reorder,ind_list,ntrain
    
def corr_vals_stoch(Q_syn,m,nbasins):
    """ Calculates the cross correlation coefficient across the basins for all stochastic realizations.
        Q: input numpy array containing runoff values
        m: # stochastic realizations
        nbasins: # basins."""
    corr_vals_stoch = np.zeros([nbasins,nbasins,m])
    for k in range(m):
        corr_vals_stoch[:,:,k] = np.corrcoef(Q_syn[:,:,k].T) #get the correlation coefficients for each stochastic realization
    return corr_vals_stoch
    
def generate_xml(q_det,Qf,m,id_list,name_list,fpath,spath):
    """This function generates CSVs for m stochastic realizations of runoff.
    Inputs: q_det, deterministic xanthos historical runoff nyears x nbasins
            Qf: matrix of stochastic future runoff, nfyears x m x nbasins
            m: # realizations to generate XMLs for
            id_list: list of basin IDs
            name_list: list of basin names
            fpath: location of mapping files (for basin naming conventions)
            spath: location to save resulting CSV files
    Returns: saves CSV that can be easily made into XML using csv_to_xml.R"""
    for i in range(m):    #for a given realization
        Q_mod = np.concatenate((q_det[nyears-5:,:],Qf[:,:,i]))#combine deterministic historical last 5 years with stochastic runoff
        Q_ma = pd.DataFrame(Q_mod).rolling(5).mean() #calculate backwards rolling mean
        Q_ma = Q_ma.iloc[5:,:] #start with value in 2020
        Q_ma = Q_ma.to_numpy()
        conv_Q = pd.DataFrame() #empty dataframe
        conv_Q['basin_id'] = id_list
        conv_Q['basin_name'] = name_list
        Q_scen = pd.DataFrame(Q_ma[::5,:]) #select every 5 years from rolling mean
        Q_scen = Q_scen.T
        conv_Q = pd.concat([conv_Q,Q_scen],axis=1) #add to larger matrix
        # create list of column names for the years
        yrs = np.linspace(2020,2100,17).astype(int)
        col_names = conv_Q.columns[2:].to_numpy()
        col_dict = dict(zip(col_names,yrs))
        conv_Q = conv_Q.rename(columns=col_dict) #rename columns
        conv_Q = conv_Q.melt(id_vars=['basin_id','basin_name'],var_name='year',
                            value_name = 'maxSubResource') # restructure dataframe to appropriate format for xml
        #load in mapping CSVs - this is so we are assigning the correct region
        basinid_gluname = pd.read_csv(fpath+'basin_to_country_mapping.csv')
        gluname_region = pd.read_csv(fpath+'basin_to_region_mapping.csv')
        #make dictionaries
        bname_dict = dict(zip(basinid_gluname.GCAM_basin_ID,
                             basinid_gluname.GLU_name))
        basreg_dict = dict(zip(gluname_region.gcam_basin_name,
                               gluname_region.region))
        conv_Q['renewresource'] = conv_Q['basin_id'].map(bname_dict)
        conv_Q['region'] = conv_Q['renewresource'].map(basreg_dict)
        conv_Q['sub.renewable.resource'] = 'runoff'
        conv_Q['renewresource'] = conv_Q['renewresource'] + '_water withdrawals'
        conv_Q = conv_Q[~conv_Q.region.isna()]
        conv_Q = conv_Q.sort_values(by=['basin_id','year'])
        conv_Q = conv_Q.filter(['region','renewresource','sub.renewable.resource','year','maxSubResource'])
        #conv_Q.set_index('region').to_csv(spath+'stochastic_runoff_'+str(i+1)+'.csv')
