#import statements
import numpy as np
import statsmodels.api as sm

def future_stoch_mvnorm(df,df_fut,m):
    """ This function applies a stochastic error model to a dataset of annual runoff in multiple locations,
        assumes historical error distribution applies to future as well. 
        Inputs:
        df = input dataframe containing annual runoff data for "true" and "model"
        df_fut = input dataframe containing annual runoff data for model projection
        m = # stochastic realizations to generate
        
        Outputs:
        Q_syn: numpy array containing stochastic realizations of annual runoff in both historical and future periods
        D_syn: numpy array containing stochastic realizations of errors in annual runoff in both historical and future periods
        res_list: model fit for linear regression of observed errors versus modeled runoff in historical period"""
    basin_nms = df['basin_name'].unique()
    nyears = len(df['year'].unique()) #time periods (every year)
    nbasins = len(basin_nms)
    nfyears = len(df_fut.year.unique()) #how many future time periods (every 5 years)

    #numpy arrays for observed, deterministic model, and difference (observed minus deterministic model)

    q_det = np.reshape(df.groupby(['basin_name'],group_keys=False)['q_det'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    q_obs = np.reshape(df.groupby(['basin_name'],group_keys=False)['q_obs'].apply(lambda x:x).to_numpy(),
                       (nyears,nbasins),order='F')
    qfut_sim = np.reshape(df_fut.groupby(['basin_name'],group_keys=False)['q_fut'].apply(lambda x:x).to_numpy(),
                       (nfyears,nbasins),order='F')

    #empty storage arrays:
    D_syn = np.zeros([nyears+nfyears,nbasins,m])
    Q_syn = np.zeros([nyears+nfyears,nbasins,m])
    res_list = []
    
    #matrix for storing mean vector and standard deviation of prediction vector
    yfit_stor = np.zeros([nyears+nfyears,nbasins])
    e_stor = np.zeros([nyears,nbasins])
    sdp_stor = np.zeros([nyears+nfyears,nbasins])
    
    for j in range(nbasins): #for each basin
        boi = df[df.basin_name==basin_nms[j]]
        boi_fut = df_fut[df_fut.basin_name==basin_nms[j]]
        q_obs_basin = boi.q_obs.to_numpy()
        q_det_basin = boi.q_det.to_numpy() 
        qfut_sim_basin = boi_fut.q_fut.to_numpy()
        diff_basin = q_obs_basin - q_det_basin

        #fit OLS model
        X_basin = sm.add_constant(q_det_basin) #add column of ones to array of independent variable
        
        mod = sm.OLS(diff_basin,X_basin) #fit ordinary least squares model, first dependent variable than independent variable
        results = mod.fit() #fit model

        #now calculate the variance of prediction
        beta = np.matmul(np.linalg.inv(np.matmul(X_basin.T,X_basin)),np.matmul(X_basin.T,diff_basin)) #these are the beta coefficients for OLS Y = Xb

        yfit = np.matmul(X_basin,beta)
        e = yfit - diff_basin
        sige = np.std(e) #standard deviation of fitted model minus D

        #calculate variance of prediction
        vare = sige**2
        varyhat = vare * np.matmul(np.matmul(X_basin,np.linalg.inv(np.matmul(X_basin.T,X_basin))),X_basin.T)
        varypred = np.diag(varyhat+vare) #take diagonal elements of matrix
        
        #NOW CALCULATE THE FUTURE YFIT
        Xfut_basin = sm.add_constant(qfut_sim_basin)
        yfit_fut = np.matmul(Xfut_basin,beta)   #Xfut*beta coefficients --> yfit future

        # #calculate variance of prediction
        varyhat_fut = vare*np.diag(np.matmul(np.matmul(Xfut_basin,np.linalg.inv(np.matmul(X_basin.T,X_basin))),Xfut_basin.T))
        varypred_fut = varyhat_fut+vare #take diagonal elements of matrix
        varypred_fut = np.array(varypred_fut,dtype='float')
        
        #save the values
        yfit_stor[:,j] = np.concatenate((yfit,yfit_fut))
        e_stor[:,j] = e
        sdp_stor[:,j] = np.concatenate((np.sqrt(varypred),np.sqrt(varypred_fut)))
        res_list.append(results)
    
    corr_e = np.corrcoef(e_stor.T)
    for i in range(nyears+nfyears):
        #get mean vector for year i
        yfit_vals = yfit_stor[i,:]
        
        #build the covariance matrix for year i
        sd_vals = sdp_stor[i,:]
        sd_matrix = np.multiply.outer(sd_vals,sd_vals)
        cov_matrix = np.multiply(corr_e,sd_matrix)
            
        # now generate stochastic realizations, loop through time periods
        #generate errors from multivariate normal distribution
        D_syn[i,:,:] = np.random.multivariate_normal(yfit_vals,
                                                     cov_matrix,
                                                     m).T
        #qobs* = qsyn = qdet + D*
        qdet_fut = np.concatenate((q_det,qfut_sim)) #add future deterministic to historical deterministic
        Q_syn[i,:,:] = np.tile(qdet_fut[i,:],(m,1)).T+D_syn[i,:,:]
        
    return Q_syn,D_syn,res_list
