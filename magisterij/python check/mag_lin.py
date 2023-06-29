import matplotlib.pyplot as plt
import numpy as np
import GPy
import deepgp
import theano
import time
import scipy.io
import createsplitlin as createsplit
import plotgp as pgp
import pandas as pd


#data='MJexample'
#X,Y,N,Ntr=createsplit.create_training_data(data)

#data='silverbox'
#data='weather'
data='weather_years'
#number of inducing points
# Perform 5 random trials
for j in range(1):
    error_GP = []
    error_DGP = []
    print('# Running repeat: ', str(j))
    Xtr,Ytr,Xts,Yts = createsplit.create_training_data(data)
    Nind=30
#     GP baseline 
    GP_time=time.time()
    kern= GPy.kern.RBF(Xtr.shape[1], ARD=1)
    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr, kernel=kern, num_inducing=Nind)
##    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr,kernel=kern, num_inducing=Nind)
#    #    mGP = GPy.models.SparseGPRegression(X = Xtr, Y = Ytr, Z = np.full((Nind, Xtr.shape[1]), 0))
#    #    mGP = GPy.models.SparseGPRegression(X = Xtr, Y = Ytr, Z = np.repeat(np.linspace(-1,1,Nind)[:,None],Xtr.shape[1],axis=1))
#    mGP.optimize(max_iters=4000,  messages=0)
#    pred_GP,var_GP = mGP.predict(Xts)
#    Nind=100
#    GP_time=time.time()
#    kern= GPy.kern.RBF(Xtr.shape[1], ARD=1)
#    kern += GPy.kern.White(Xtr.shape[1])
#     Bayesian Gaussian Process Latent Variable Model with Descriminative prior
#    mGP=GPy.models.BayesianGPLVM(Ytr,Xtr.shape[1],X=Xtr,kernel=kern,num_inducing=Nind)
###    mGP.latent_space.mean=np.repeat(np.linspace(-np.random.rand(),np.random.rand(),Xtr.shape[0])[:,None],6,axis=1)
###    mGP.inducing_inputs=np.hstack((np.linspace(np.min(Xtr[:,0]),np.max(Xtr[:,0]),Nind)[:,None],np.linspace(np.min(Xtr[:,1]),np.max(Xtr[:,1]),Nind)[:,None],np.linspace(np.min(Xtr[:,2]),np.max(Xtr[:,2]),Nind)[:,None],
###                                   np.linspace(np.min(Xtr[:,3]),np.max(Xtr[:,3]),Nind)[:,None],np.linspace(np.min(Xtr[:,4]),np.max(Xtr[:,4]),Nind)[:,None],np.linspace(np.min(Xtr[:,5]),np.max(Xtr[:,5]),Nind)[:,None]))
##    mGP.sum.rbf.variance=np.random.rand()*100
##    for i in range(Xtr.shape[1]):
##        mGP.sum.rbf.lengthscale[i]=np.random.rand()*100
##    mGP.Gaussian_noise.variance=np.random.rand()/1000
    mGP.optimize(max_iters=4000,  messages=0)
    pred_GP,var_GP = mGP.predict(Xts)
    GP_time=time.time()-GP_time
    print'I am done with the GP. It took me '+str(GP_time)+' seconds.'
#    Nind=10
#    # DGP baseline
#    dGP_time=time.time()
#    Q1=1
#    m = deepgp.DeepGP([Ytr.shape[1],Q1,Xtr.shape[1]],Ytr, X=Xtr,kernels=[GPy.kern.RBF(Q1),GPy.kern.RBF(Xtr.shape[1])], num_inducing=Nind, back_constraint=False)
#    m.obslayer.latent_space.mean = np.linspace(0.0,2.0,Xtr.shape[0])[:,None]
#    m.inducing_inputs=np.linspace(0.0,3.0,Nind)[:,None]
#    m.obslayer.rbf.variance=0.7
#    m.obslayer.rbf.lengthscale=15
#    m.obslayer.Gaussian_noise.variance = 0.0005
#    for i in range(len(m.layers)):
#        if i == 0:
#            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.var()*0.01
#        else:
#            m.layers[i].Gaussian_noise.variance = m.layers[i].Y.mean.var()*0.005
#        # Fix noise for a few iters, so that it avoids the trivial local minimum to only learn noise.
#        m.layers[i].Gaussian_noise.variance.fix()
#    
#    m.optimize(max_iters=200, messages=0)
#    # Now unfix noise and learn normally.
#    for i in range(len(m.layers)):
#        m.layers[i].Gaussian_noise.variance.unfix()
#    m.optimize(max_iters=4000, messages=0)
#    
#    pred,var = m.predict(Xts)
#    dGP_time = time.time()-dGP_time
#    
##    print m
#    
#    error_DGP.append(createsplit.nrmse(pred, Yts))
    error_GP.append(createsplit.nrmse(pred_GP, Yts))
#    
    error_GP = np.array(error_GP)
#    error_DGP = np.array(error_DGP)
#    
    varGP=2*np.mean(np.sqrt(var_GP))
#    varDGP=2*np.mean(np.sqrt(var))
#    
    print'# Error GP:  '+ str(error_GP.mean())+ ' with std: '+ str(varGP)
#    print'# Error DGP: '+ str(error_DGP.mean())+ ' with std: '+ str(varDGP)
#    
    print'# GP_time = '+str(GP_time)+' seconds'+' for j='+str(j)
#    print'# dGP_time = '+str(dGP_time)+' seconds'+' for j='+str(j)
    
    print '================ARD values========================'
    print str(np.atleast_2d(mGP.kern.input_sensitivity(summarize=False)))
    print '================ARD values========================'
#    save predictions and variances to .txt file 
#    tmp = np.concatenate((pred,var,pred_GP,var_GP,Yts),axis=1)
#    names = ['pred DGP','var DGP','pred_GP','var_GP','Test']
#    tmp = pd.DataFrame(tmp, columns=names)
#    np.savetxt('results_{}.txt'.format(j), tmp.values, delimiter="\t") 
#    print 'Saving successful. Results are saved in results'+str(j)+'.txt file.'
#    if varDGP<var_0:
#        var_0=varDGP
#        tmp = np.concatenate((pred,var,Yts),axis=1)
#        names = ['pred DGP','var DGP','Test']
#        tmp = pd.DataFrame(tmp, columns=names)
#        np.savetxt('results.txt', tmp.values, delimiter="\t") 
#        print 'Saving successful. Results are saved in results'+str(j)+'.txt file.'
#        
#        with open('model_prop.txt', 'w') as file:
#            file.writelines(str(m))
#            file.writelines(str(m.obslayer.rbf.lengthscale))
#            file.writelines(str(m.obslayer.rbf.variance))
#            file.writelines(str(m.layer_1.rbf.lengthscale))
#            file.writelines(str(m.layer_1.rbf.variance))
#        file.writelines(str(m))
#        var_0=varDGP
    
    #save other technical data to .txt
#    with open('resluts_details_{}.txt'.format(j), 'w') as file:
#        file.writelines('Script run on '+str(time.strftime("%d/%m/%Y"))+' at '+str(time.strftime("%H:%M:%S"))+'\n')
#        file.writelines('# Error GP:  '+ str(error_GP.mean())+ ' with std: '+ str(error_GP.std())+'\n')
#        file.writelines('# Error DGP: '+ str(error_DGP.mean())+ ' with std: '+ str(error_DGP.std())+'\n')
#        file.writelines('# GP_time = '+str(GP_time)+' seconds \n')
#        file.writelines('# dGP_time = '+str(dGP_time)+' seconds \n')
    
#    with open('model_prop_{}.txt'.format(j), 'w') as file:
#        file.writelines(str(m))
#        file.writelines(str(mGP))

#plt.plot(pred,'o-b',label='Deep GP')
#plt.plot(pred_GP,'x-r',label='GP')
#plt.plot(Yts,'+k--',label='Test')
#plt.legend()
#plt.xlabel('k [Integer]')
#plt.ylabel('Output')
#plt.show()

##save predictions and variances to .txt file 
#tmp = np.concatenate((pred,var,pred_GP,var_GP,Yts),axis=1)
#names = ['pred DGP','var DGP','pred_GP','var_GP','Test']
#tmp = pd.DataFrame(tmp, columns=names)
#np.savetxt('results.txt', tmp.values, delimiter="\t") 
#print 'Saving successful. Results are saved in results.txt file.'
#
##save other technical data to .txt
#with open('resluts_details.txt', 'w') as file:
#    file.writelines('Script run on '+str(time.strftime("%d/%m/%Y"))+' at '+str(time.strftime("%H:%M:%S"))+'\n')
#    file.writelines('# Error GP:  '+ str(error_GP.mean())+ ' with std: '+ str(error_GP.std())+'\n')
#    file.writelines('# Error DGP: '+ str(error_DGP.mean())+ ' with std: '+ str(error_DGP.std())+'\n')
#    file.writelines('# GP_time = '+str(GP_time)+' seconds \n')
#    file.writelines('# dGP_time = '+str(dGP_time)+' seconds \n')

#t=np.arange(1,len(Yts)+1,1)
#pgp.plotgp(1,t,Yts,pred_GP.T[0],np.sqrt(var_GP.T[0]))
#pgp.plotgp(2,t,Yts,pred.T[0],np.sqrt(var.T[0]))