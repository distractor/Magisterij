import matplotlib.pyplot as plt
import numpy as np
import GPy
import deepgp
import time
import scipy.io
import pandas as pd

import sys
sys.path.append("C:/Users/mitja/Documents/magisterij/python check/createsplit")
import createsplit
sys.path.append("C:/Users/mitja/Documents/magisterij/python check")
import plotgp as pgp

filepath='C:/Users/mitja/Documents/magisterij/python check/'

#define dataset
#data='silverbox'
data='weather_years'

# Perform 5 random trials
for j in range(1):
    #inicializacija vektorjev NRMSE napak
    error_GP = []
    error_DGP = []
    print('# Running repeat: ', str(j))
    #inicializacija testnih in učnih vrednosti
    Xtr,Ytr,Xts,Yts,Xstd,Xmean = createsplit.create_training_data(data)
    #number of inducing points
    Nind=1
    GP_time=time.time()
    """
    Stvari ki se jih da nastavit vidimo tako, da v konzolo vpišemo "print(mGP)"
    """
    kern = GPy.kern.RBF(Xtr.shape[1], ARD=1)
    kern += GPy.kern.White(Xtr.shape[1]) + GPy.kern.Bias(Xtr.shape[1])
    """
    GPFITC
    """
#    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr, kernel=kern, num_inducing=Nind)
    """
    Bayesian Gaussian Process Latent Variable Model with Descriminative prior
    """
    mGP=GPy.models.BayesianGPLVM(Ytr,Xtr.shape[1],X=Xtr,kernel=kern,num_inducing=Nind)
#    mGP.latent_space.mean=np.repeat(np.linspace(0.0,0.7,Xtr.shape[0])[:,None],6,axis=1)
#    mGP.inducing_inputs=np.hstack((np.linspace(np.min(Xtr[:,0]),np.max(Xtr[:,0]),Nind)[:,None],np.linspace(np.min(Xtr[:,1]),np.max(Xtr[:,1]),Nind)[:,None],np.linspace(np.min(Xtr[:,2]),np.max(Xtr[:,2]),Nind)[:,None],
#                                   np.linspace(np.min(Xtr[:,3]),np.max(Xtr[:,3]),Nind)[:,None],np.linspace(np.min(Xtr[:,4]),np.max(Xtr[:,4]),Nind)[:,None],np.linspace(np.min(Xtr[:,5]),np.max(Xtr[:,5]),Nind)[:,None]))
#    mGP.Gaussian_noise.variance=0.0001
    mGP.optimize(max_iters=4000,  messages=0)
    pred_GP,var_GP = mGP.predict(Xts)
    GP_time=time.time()-GP_time

    #za izris ARD
#    mGP.kern.plot_ARD(legend=True)
    """ 
    numerične vrednosti ARD:
            print '================ARD values========================'
            print str(np.atleast_2d(mGP.kern.input_sensitivity(summarize=False)))
            print '================ARD values========================'
    """
    print'I am done with the GP. It took me '+str(GP_time)+' seconds. Now starting with deepGP.'
    #number of inducing points
    Nind=10
    # DGP baseline
    dGP_time=time.time()
    #št nivojev Q1
    Q1=1
    kern= GPy.kern.RBF(Xtr.shape[1], ARD=1)
    kern += GPy.kern.White(Xtr.shape[1])
    #inicializacija deepGP
    m = deepgp.DeepGP([Ytr.shape[1],Q1,Xtr.shape[1]],Ytr, X=Xtr,kernels=[GPy.kern.RBF(Q1),GPy.kern.RBF(Xtr.shape[1])], num_inducing=Nind, back_constraint=False)
    """
    deep gp s 3 nivoji:
        Q=1,2,3
        m=deepgp.DeepGP([Ytr.shape[1],Q[0],Q[1],Q[2],Xtr.shape[1]],Ytr, X=Xtr,kernels=[GPy.kern.RBF(Q[0]),GPy.kern.RBF(Q[1]),GPy.kern.RBF(Q[2]),GPy.kern.RBF(Xtr.shape[1])], num_inducing=Nind, back_constraint=False)
    """
#    m.inducing_inputs=np.linspace(0.0,1.0,Nind)[:,None]
#    m.obslayer.rbf.variance=2
#    m.obslayer.rbf.lengthscale=20
#    m.obslayer.Gaussian_noise.variance = 0.001 
#    m.layer_1.rbf.variance =0.7 #default 1
#    m.layer_1.rbf.lengthscale = 30 #default 1
#    m.layer_1.Gaussian_noise.variance=0.0005
#    m.layer_1.inducing_inputs=np.repeat(np.linspace(-1.0,1.0,Nind)[:,None],1,axis=1)
#    m.layer_1.latent_space.mean=np.repeat(np.linspace(-2.0,2.0,Xtr.shape[0])[:,None],2,axis=1)
    m.obslayer.latent_space.mean = np.linspace(0.0,1.0,Xtr.shape[0])[:,None]
    # Here we initialize such that Signal to noise ratio is high.
    for i in range(len(m.layers)):
        if i == 0:
            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.var()*0.01
        else:
            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.mean.var()*0.005
        # Fix noise for a few iters, so that it avoids the trivial local minimum to only learn noise.
        m.layers[i].Gaussian_noise.variance.fix()

    m.optimize(max_iters=200, messages=0)
    # Now unfix noise and learn normally.
    for i in range(len(m.layers)):
        m.layers[i].Gaussian_noise.variance.unfix()
    m.optimize(max_iters=4000, messages=0)
    #get model prediction and variance
    pred,var = m.predict(Xts)
    dGP_time = time.time()-dGP_time
    #check parameter values:
#    print m
    #izračun NRMSE
    error_DGP.append(createsplit.nrmse(pred, Yts))
    error_GP.append(createsplit.nrmse(pred_GP, Yts))
    error_GP = np.array(error_GP)
    error_DGP = np.array(error_DGP)
    
    varGP=2*np.mean(np.sqrt(var_GP))
    varDGP=2*np.mean(np.sqrt(var))
    
    print'# Error GP:  '+ str(error_GP.mean())+ ' with confidence interval: '+ str(varGP)
    print'# Error DGP: '+ str(error_DGP.mean())+ ' with confidence interval: '+ str(varDGP)
    
    print'# GP_time = '+str(GP_time)+' seconds'+' for j='+str(j)
    print'# dGP_time = '+str(dGP_time)+' seconds'+' for j='+str(j)
    
    
#    tmp = np.concatenate((pred,var,pred_GP,var_GP,Yts),axis=1)
#    names = ['pred DGP','var DGP','pred_GP','var_GP','Test']
#    tmp = pd.DataFrame(tmp, columns=names)
#    np.savetxt(filepath+'resultsARD_{}.txt'.format(j), tmp.values, delimiter="\t") 
#    print 'Saving successful. Results are saved in results.txt file.'

    
    t=np.arange(1,len(Yts)+1,1)
    pgp.plotgp(2,t,Yts,pred_GP.T[0],np.sqrt(var_GP.T[0]),partly=False)
    pgp.plotgp(j+3,t,Yts,pred.T[0],np.sqrt(var.T[0]),partly=False)
