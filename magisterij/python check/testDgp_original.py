
from matplotlib import pylab as plt
plt.ion()
import numpy as np
# from: https://github.com/SheffieldML/GPy
import GPy

# from: https://github.com/SheffieldML/PyDeepGP
import deepgp

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

N=300
Ntr = 40
X = np.linspace(0,1,N)
Xmean = X.mean()
Xstd = X.std()

X-=Xmean
X/=Xstd

Y = []
fac=4
for i in range(N):
    tmp = np.sin(fac*X[i]) * np.cos(fac*X[i]*X[i])
    Y.append(tmp)

Y=np.array(Y)[:,None]
X = X[:,None]

Yorig = Y.copy()

error_GP = []
error_DGP = []

# Perform 5 random trials
for j in range(1):
    print('# Running repeat: ', str(j))
    Y = Yorig.copy()
    Y+= np.random.randn(Y.shape[0])[:,None]* 0.05

    # split in train and test 
    perm = np.random.permutation(N)
    index_tr = np.sort(perm[0:Ntr])
    index_ts = np.sort(perm[Ntr:])
    Xtr = X[index_tr,:]
    Ytr = Y[index_tr,:]
    Xts = X[index_ts,:]
    Yts = Y[index_ts,:]

    # GP baseline 
    mGP = GPy.models.SparseGPRegression(X=Xtr, Y=Ytr, num_inducing=Ntr)
    mGP.optimize(max_iters=4000,  messages=0)
    pred_GP = mGP.predict(Xts)[0]

    # DGP baseline
    m = deepgp.DeepGP([Ytr.shape[1],1,Xtr.shape[1]],Ytr, X=Xtr,kernels=[GPy.kern.RBF(1), GPy.kern.RBF(X.shape[1])], num_inducing=Ntr, back_constraint=False)
    # For so simple 1D data, we initialize the middle layer (latent space) to be the input.
    m.obslayer.X.mean = Xtr.copy()
    # We can initialize and then fix the middle layer inducing points to also be the input. 
    # Another strategy would be to (also/or) do that in the top layer.
    m.obslayer.Z[:] = Xtr[:].copy()
    m.obslayer.Z.fix()
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

    pred = m.predict(Xts)[0]

    # plt.plot(Xts, pred_GP, 'x-r', label='GP')
    # plt.plot(Xts, pred, 'o-b', label='DGP')
    # plt.plot(Xts, Yts, '+k--', label='True')
    # plt.legend()


    error_DGP.append(rmse(pred, Yts))
    error_GP.append(rmse(pred_GP, Yts))

error_GP = np.array(error_GP)
error_DGP = np.array(error_DGP)
print('# Error GP:  ', error_GP.mean(), ' with std: ', error_GP.std())
print('# Error DGP: ', error_DGP.mean(), ' with std: ', error_DGP.std())

plt.plot(pred,'o-b',label='Deep GP')
plt.plot(pred_GP,'x-r',label='GP')
plt.plot(Yts,'+k--',label='Test')
plt.legend()
plt.xlabel('k [Integer]')
plt.ylabel('Output')
plt.show()