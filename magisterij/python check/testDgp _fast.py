import matplotlib.pyplot as plt
import numpy as np
import GPy
import deepgp
import time

import sys
sys.path.append("C:/Users/mitja/Documents/magisterij/python check/createsplit")
import faster as cr
import plotgp as pgp

#data='MJexample'
#X,Y,N,Ntr=createsplit.create_training_data(data)

data='silverbox'
X,Y=cr.create_training_data(data)

#data='weather'
#X,Y = cr.create_training_data(data)

#number of inducing points
Nind=100
#rescale
X=cr.rescale(X)
Y=cr.rescale(Y)

error_GP = []
error_DGP = []

# Perform 5 random trials
for j in range(1):
    print('# Running repeat: ', str(j))
    Xtr,Ytr,Xts,Yts=cr.split_data(data,X,Y,N=1,Ntr=1)
    Xts = cr.rescale(Xts)
    Yts = cr.rescale(Yts)
        
    # GP baseline 
    GP_time=time.time()
    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr, num_inducing=Nind)
    #mGP = GPy.models.SparseGPRegression(X = Xtr, Y = Ytr, Z = np.full((Nind, X.shape[1]), 0))
    mGP.optimize(max_iters=4000,  messages=0)
    pred_GP,var_GP = mGP.predict(Xts)
    GP_time=time.time()-GP_time
    
    # DGP baseline
    dGP_time=time.time()
    m = deepgp.DeepGP([Ytr.shape[1],1,Xtr.shape[1]],Ytr, X=Xtr,kernels=[GPy.kern.RBF(1), GPy.kern.RBF(X.shape[1])], num_inducing=Nind, back_constraint=False)
    # For so simple 1D data, we initialize the middle layer (latent space) to be the input.
    #m.obslayer.X.mean = Xtr.copy()
    # We can initialize and then fix the middle layer inducing points to also be the input. 
    # Another strategy would be to (also/or) do that in the top layer.
    #m.obslayer.Z[:] = Xtr[:].copy()
    #m.obslayer.Z.fix()
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

    pred,var = m.predict(Xts)
    dGP_time = time.time()-dGP_time

#    for col in range(6):
#        plt.plot(Xts[:,col], pred_GP, 'x-r', label='GP')
#        plt.plot(Xts[:,col], pred, 'o-b', label='DGP')
#        plt.plot(Xts[:,col], Yts, '+k--', label='True')
#        plt.title('col'+str(col))
#        plt.legend()
#        plt.show()
#    
#        
    error_DGP.append(cr.nrmse(pred, Yts))
    error_GP.append(cr.nrmse(pred_GP, Yts))
    
    
error_GP = np.array(error_GP)
error_DGP = np.array(error_DGP)
print'# Error GP:  '+ str(error_GP.mean())+ ' with std: '+ str(error_GP.std())
print'# Error DGP: '+ str(error_DGP.mean())+ ' with std: '+ str(error_DGP.std())
print'# GP_time = '+str(GP_time)+' seconds'
print'# dGP_time = '+str(dGP_time)+' seconds'

#plt.plot(pred,'o-b',label='Deep GP')
#plt.plot(pred_GP,'x-r',label='GP')
#plt.plot(Yts,'+k--',label='Test')
#plt.legend()
#plt.xlabel('k [Integer]')
#plt.ylabel('Output')
#plt.show()

t=np.arange(1,len(Yts)+1,1)
pgp.plotgp(1,t,Yts,pred_GP.T[0],np.sqrt(var_GP.T[0]))
