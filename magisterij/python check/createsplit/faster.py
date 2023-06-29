# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 11:08:27 2017

Created to make the code more readable.

@author: Mitja Jancic
"""
import numpy as np
import scipy.io
import pandas as pd
import math

def nrmse(predictions, targets):
    """
    should be same as goodnessOfFit in Matlab: https://www.mathworks.com/help/ident/ref/goodnessoffit.html
    """
    list=[]
    for dim in range(targets.shape[0]):
        tmp=1-np.linalg.norm(targets-predictions,2)/np.linalg.norm(targets-np.mean(targets),2)
        list.append(tmp)
    return list
def f7(seq):
    """
    removes duplicated items while preserving order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
def remove_nan(X,Y):
    """
    Now remove redundant values 'nan'
    """
    bad_index_X = []
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if math.isnan(X[row,col]):
                bad_index_X.append(row)
                break
    bad_index_Y = []
    for row in range(Y.shape[0]):
        for col in range(Y.shape[1]):
            if math.isnan(Y[row,col]):
                bad_index_Y.append(row)
                break
    
    bad_index = bad_index_X+bad_index_Y
    bad_index = f7(bad_index)
    X=np.delete(X,bad_index, axis = 0)
    Y=np.delete(Y,bad_index, axis = 0)
    
    return X,Y

def create_training_data(data_type):
    """
    creates training data
    """
    if data_type=='silverbox':
        silverbox_data = scipy.io.loadmat('C:/Users/mitja/Documents/magisterij/sinteticni/sb.mat')
        #merge training and test data
        sb_train = silverbox_data['data_train']
        #sb_test = silverbox_data['data_test']
        #silverbox_data=np.concatenate((sb_train,sb_test),axis=0)
        #rescale
        #silverbox_data=rescale(silverbox_data)
        #split data back to train and test
        #sb_train = silverbox_data[:10000]
        #sb_test = silverbox_data[10000:]
        
        #define training regresors
        _Xtr=np.concatenate((sb_train[0:-4,1],sb_train[1:-3,1],sb_train[2:-2,1],sb_train[3:-1,1],
                           sb_train[0:-4,0],sb_train[1:-3,0],sb_train[2:-2,0],sb_train[3:-1,0]),axis=0).reshape(8,len(sb_train[0:-4,1])).T
        _Ytr=sb_train[4:,1,None]
        return _Xtr,_Ytr
    elif data_type=='MJexample':
        N=300
        Ntr = 150
        X = np.linspace(0,1,N)
        X = np.array([X,X*X,np.linspace(5,10,N),np.sin(X),np.cos(X),np.sin(X)*np.cos(X)]).T
        
        fac=10
        Y=np.array([np.sin(fac*x)*np.cos(fac*x**2) for x in X[:,0]])[:,None]
        #Y=np.asarray([np.log(np.linspace(1,100,N)),np.cos(np.linspace(0,100,N))]).T
        _X=X
        _Y=Y
        return _X,_Y,N,Ntr
    elif data_type=='weather':
        """
        creates a regressor 
        X=[T(k-1),vl(k-1),vl(k),St(k-1),St(k),Sev(k-3),Sev(k-2),Sev(k-1),Sev(k),p(k-1),p(k),Ve(k-1),Ve(k)]
        Y=[T(k)]
        """
        filepath='C:/Users/mitja/Documents/magisterij/realni/podatki.txt'
        data = pd.read_csv(filepath, header=0 ,delimiter='\t').values
        #remove first two columns
        data = data[0:,1:].astype(float)
        #rescale to zero mean and std 1
        #data = rescale(data)
        #formulate regressor
        _X = np.concatenate((data[2:-1,0],data[2:-1,1],data[3:,1],data[2:-1,2],data[3:,2],
                             data[:-3,3],data[1:-2,3],data[2:-1,3],data[3:,3],
                             data[2:-1,4],data[3:,4],data[2:-1,5],data[3:,5]),axis=0).reshape(13,len(data[3:,3])).T
#        _X = np.concatenate((data[2:-1,2],data[3:,2],
#                             data[2:-1,5],data[3:,5]),axis=0).reshape(4,len(data[3:,3])).T
        _Y = data[2:-1,0,None]
        #remove redundant regressors with 'nan' values
        _X,_Y=remove_nan(_X,_Y)
        #delete every 3rd regressor from data
        _Xtr = np.delete(_X,np.arange(0,_X.size,5),axis=0)
        _Ytr = np.delete(_Y,np.arange(0,_X.size,5),axis=0)
        return _Xtr,_Ytr,
    
def split_data(data_type,X,Y,N,Ntr):
    """
    splits data into training and test
    """
    if data_type=='silverbox':
        #define test regresors
        silverbox_data = scipy.io.loadmat('C:/Users/mitja/Documents/magisterij/sinteticni/sb.mat')
        #merge training and test data
        sb_test = silverbox_data['data_test']
        _Xts=np.concatenate((sb_test[0:-4,1],sb_test[1:-3,1],sb_test[2:-2,1],sb_test[3:-1,1],
                           sb_test[0:-4,0],sb_test[1:-3,0],sb_test[2:-2,0],sb_test[3:-1,0]),axis=0).reshape(8,len(sb_test[0:-4,1])).T
        _Yts=sb_test[4:,1,None]
        #add noise
        Y+= np.random.randn(Y.shape[0])[:,None]* 0.01
        _Yts+= np.random.randn(_Yts.shape[0])[:,None]* 0.01
        Xtr=X
        Ytr=Y
        Xts=_Xts
        Yts=_Yts
        
#        Xtr=X[0:500,]
#        Ytr=Y[0:500,]
#        Xts=_Xts[0:100,]
#        Yts=_Yts[0:100,]
    elif data_type=='MJexample':
        #Y+= np.random.randn(Y.shape[0])[:,None]* 0.05
    
        # split in train and test 
        perm = np.random.permutation(N)
        index_tr = np.sort(perm[0:Ntr])
        index_ts = np.sort(perm[Ntr:])
        Xtr = X[index_tr,:]
        Ytr = Y[index_tr,:]
        Xts = X[index_ts,:]
        Yts = Y[index_ts,:]
    elif data_type=='weather':
        #import values
        filepath='C:/Users/mitja/Documents/magisterij/realni/podatki.txt'
        data = pd.read_csv(filepath, header=0 ,delimiter='\t').values
        #remove first two columns
        data = data[0:,1:].astype(float)
        #rescale to zero mean and std 1
        #data = rescale(data)
        #formulate regressor
        _X = np.concatenate((data[2:-1,0],data[2:-1,1],data[3:,1],data[2:-1,2],data[3:,2],
                             data[:-3,3],data[1:-2,3],data[2:-1,3],data[3:,3],
                             data[2:-1,4],data[3:,4],data[2:-1,5],data[3:,5]),axis=0).reshape(13,len(data[3:,3])).T
#        _X = np.concatenate((data[2:-1,2],data[3:,2],
#                             data[2:-1,5],data[3:,5]),axis=0).reshape(4,len(data[3:,3])).T                     
        _Y = data[2:-1,0,None]
        #add noise
        _Y+= np.random.randn(_Y.shape[0])[:,None]* 0.05
        Y+= np.random.randn(Y.shape[0])[:,None]* 0.05
        #remove redundant regressors with 'nan' values
        _X,_Y=remove_nan(_X,_Y)
        #collect only every third regressor from data to be test data
        Xts = _X[::5]
        Yts = _Y[::5]
        Xtr = X
        Ytr = Y
        
    return Xtr,Ytr,Xts,Yts

def rescale(X):
    """
    removes the average and sets standard deviation to 1
    """
    Xmean = np.mean(X,axis=0)
    X = np.subtract(X,Xmean)
    Xstd = np.std(X,axis=0)
    X = np.divide(X,Xstd)  
    return X