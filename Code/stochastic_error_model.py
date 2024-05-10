import numpy as np
import statsmodels.api as sm
from statsmodels.stats.correlation_tools import cov_nearest
import scipy.stats as st

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
    sym = []
    mineig_orig = np.zeros([nyears+nfyears,1])
    mineig_new = np.zeros([nyears+nfyears,1])
    pct_fro = np.zeros([nyears+nfyears,1])

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
        cov_matrix_orig = np.multiply(corr_e,sd_matrix)
        
        #ensure that covariance matrix is positive definite and symmetric
        sym.append(np.allclose(cov_matrix_orig,cov_matrix_orig.T)) #symmetric?
        eigval_orig,eigvect_orig = np.linalg.eigh(cov_matrix_orig)
        mineig_orig[i] = np.min(eigval_orig) #minimum original eigenvalue
        
        cov_matrix = cov_nearest(cov_matrix_orig,threshold=10e-4) #calculate nearest covariance matrix that is positive definite
        
        eigval_new,eigvect_new = np.linalg.eigh(cov_matrix)
        mineig_new[i] = np.min(eigval_new) #minimum new eigenvalue
        fro_diff = np.linalg.norm(cov_matrix - cov_matrix_orig)  #calculate Frobenius norm of difference
        fro_orig = np.linalg.norm(cov_matrix_orig) #frobenius norm of original
        pct_fro[i] = 100*(fro_diff/fro_orig) #% of original frobenius norm contained in difference
        
        # generate errors from multivariate normal distribution (e*)
        D_syn[i,:,:] = np.random.multivariate_normal(yfit_vals,cov_matrix,m).T
        #mvn = st.multivariate_normal(yfit_vals,cov_matrix,allow_singular=False)
        #D_syn[i,:,:] = mvn.rvs(size=m).T

        # calculate stochastic ensemble of runoff
        qdet_fut = np.concatenate((q_det,qfut_sim)) #add future deterministic to historical deterministic
        Q_syn[i,:,:] = np.tile(qdet_fut[i,:],(m,1)).T+D_syn[i,:,:] # qobs* = qdet + e*
    return Q_syn,sym,mineig_orig,mineig_new,pct_fro #return stochastic runoff ensemble
