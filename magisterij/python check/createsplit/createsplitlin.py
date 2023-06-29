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
        silverbox_data = scipy.io.loadmat('/root/magisterij/sinteticni/sb.mat')
        #merge training and test data
        sb_train = silverbox_data['data_train']
        sb_test = silverbox_data['data_test']
        silverbox_data=np.concatenate((sb_train,sb_test),axis=0)
        #rescale
        silverbox_data=rescale(silverbox_data)
        #split data back to train and test
        sb_train = silverbox_data[:10000]
        sb_test = silverbox_data[10000:]
        #define training regresors
        _Xtr=np.concatenate((sb_train[0:-4,1],sb_train[2:-2,1],sb_train[3:-1,1],
                           sb_train[1:-3,0],sb_train[2:-2,0],sb_train[3:-1,0]),axis=0).reshape(6,len(sb_train[0:-4,1])).T
        _Ytr=sb_train[4:,1,None]
        
        #define test regresors
        _Xts=np.concatenate((sb_test[0:-4,1],sb_test[2:-2,1],sb_test[3:-1,1],
                           sb_test[1:-3,0],sb_test[2:-2,0],sb_test[3:-1,0]),axis=0).reshape(6,len(sb_test[0:-4,1])).T
        _Yts=sb_test[4:,1,None]
        
        #add noise
        _Yts+= np.random.randn(_Yts.shape[0])[:,None]* 0.01
        _Ytr+= np.random.randn(_Ytr.shape[0])[:,None]* 0.01
#        _Xtr=_Xtr[0:500,]
#        _Ytr=_Ytr[0:500,]
#        _Xts=_Xts[0:100,]
#        _Yts=_Yts[0:100,]
        return _Xtr,_Ytr,_Xts,_Yts
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
        filepath='/root/magisterij/realni/podatki.txt'
        data = pd.read_csv(filepath, header=0 ,delimiter='\t').values
        #remove first two columns
        data = data[0:,1:].astype(float)
        #rescale to zero mean and std 1
        data = rescale(data)
        #split to training and test data
        data_train = data[:2000,:]
        data_test = data [2000:,:]
        #define train regressors
        _Xtr = np.concatenate((data_train[2:-1,0],data_train[2:-1,1],data_train[3:,1],data_train[2:-1,2],data_train[3:,2],
                             data_train[:-3,3],data_train[1:-2,3],data_train[2:-1,3],data_train[3:,3],
                             data_train[2:-1,4],data_train[3:,4],data_train[2:-1,5],data_train[3:,5]),axis=0).reshape(13,len(data_train[3:,3])).T
        _Ytr = data_train[2:-1,0,None]
        #define test regressors
        _Xts = np.concatenate((data_test[2:-1,0],data_test[2:-1,1],data_test[3:,1],data_test[2:-1,2],data_test[3:,2],
                             data_test[:-3,3],data_test[1:-2,3],data_test[2:-1,3],data_test[3:,3],
                             data_test[2:-1,4],data_test[3:,4],data_test[2:-1,5],data_test[3:,5]),axis=0).reshape(13,len(data_test[3:,3])).T
        _Yts = data_test[2:-1,0,None]
        #remove redundant regressors with 'nan' values
        _Xtr,_Ytr=remove_nan(_Xtr,_Ytr)
        _Xts,_Yts=remove_nan(_Xts,_Yts)
        #add noise to output values
        _Ytr+= np.random.randn(_Ytr.shape[0])[:,None]* 0.01
        _Yts+= np.random.randn(_Yts.shape[0])[:,None]* 0.01
        return _Xtr,_Ytr,_Xts,_Yts
    elif data_type=='weather_years':
        data=[]
        for year in range(2013,2017):
            #we take years 2013, 2014 and 2015 for test data
            filepath='/root/magisterij/realni/podatki/stolp{}.txt'.format(year)
            data_tmp = pd.read_csv(filepath, header=0 ,delimiter='\t').values
            #remove first two columns
            data_tmp = data_tmp[0:,1:].astype(float)
            #and year 2016 for test
            if year==2016:
                test_length=len(data_tmp)
            data.append(data_tmp)
        #stack arrays 
        data=np.vstack(data)
        #rescale data to zero mean and std 1
        data = rescale(data)
        #split to training and test data
        data_train = data[:-test_length,:]
        data_test = data [-test_length:,:]
        #define train regressors
        #regresor (14 dimensions)
#        _Xtr = np.concatenate((data_train[2:-2,0],data_train[3:-1,0],
#                               data_train[2:-2,1],data_train[3:-1,1],
#                               data_train[2:-2,2],data_train[3:-1,2],
#                               data_train[0:-4,3],data_train[1:-3,3],data_train[2:-2,3],data_train[3:-1,3],
#                               data_train[2:-2,4],data_train[3:-1,4],
#                               data_train[2:-2,5],data_train[3:-1,5]),axis=0).reshape(14,len(data_train[2:-2,3])).T
#        _Ytr = data_train[4:,0,None]
#        _Xts = np.concatenate((data_test[2:-2,0],data_test[3:-1,0],
#                               data_test[2:-2,1],data_test[3:-1,1],
#                               data_test[2:-2,2],data_test[3:-1,2],
#                               data_test[0:-4,3],data_test[1:-3,3],data_test[2:-2,3],data_test[3:-1,3],
#                               data_test[2:-2,4],data_test[3:-1,4],
#                               data_test[2:-2,5],data_test[3:-1,5]),axis=0).reshape(14,len(data_test[2:-2,3])).T
#        _Yts = data_test[4:,0,None]
##        regresor (12 dimensions)
#        _Xtr = np.concatenate((data_train[2:-2,0],data_train[3:-1,0],
#                               data_train[2:-2,1],data_train[3:-1,1],
#                               data_train[2:-2,2],data_train[3:-1,2],
#                               data_train[0:-4,3],data_train[1:-3,3],data_train[2:-2,3],data_train[3:-1,3],
#                               data_train[2:-2,4],data_train[3:-1,4]
#                               ),axis=0).reshape(12,len(data_train[2:-2,3])).T
#        _Ytr = data_train[4:,0,None]
#        _Xts = np.concatenate((data_test[2:-2,0],data_test[3:-1,0],
#                               data_test[2:-2,1],data_test[3:-1,1],
#                               data_test[2:-2,2],data_test[3:-1,2],
#                               data_test[0:-4,3],data_test[1:-3,3],data_test[2:-2,3],data_test[3:-1,3],
#                               data_test[2:-2,4],data_test[3:-1,4]
#                               ),axis=0).reshape(12,len(data_test[2:-2,3])).T
#        _Yts = data_test[4:,0,None]
##        regresor (11 dimensions)
#        _Xtr = np.concatenate((data_train[2:-2,0],data_train[3:-1,0],
#                               data_train[2:-2,1],data_train[3:-1,1],
#                               data_train[2:-2,2],data_train[3:-1,2],
#                               data_train[0:-4,3],data_train[1:-3,3],data_train[2:-2,3],
#                               data_train[2:-2,4],data_train[3:-1,4]
#                               ),axis=0).reshape(11,len(data_train[2:-2,3])).T
#        _Ytr = data_train[4:,0,None]
#        _Xts = np.concatenate((data_test[2:-2,0],data_test[3:-1,0],
#                               data_test[2:-2,1],data_test[3:-1,1],
#                               data_test[2:-2,2],data_test[3:-1,2],
#                               data_test[0:-4,3],data_test[1:-3,3],data_test[2:-2,3],
#                               data_test[2:-2,4],data_test[3:-1,4]
#                               ),axis=0).reshape(11,len(data_test[2:-2,3])).T
#        _Yts = data_test[4:,0,None]
#large system:
        _Xtr = np.concatenate((data_train[0:-5,0],data_train[1:-4,0],data_train[2:-3,0],data_train[3:-2,0],data_train[4:-1,0],
                               data_train[0:-5,1],data_train[1:-4,1],data_train[2:-3,1],data_train[3:-2,1],data_train[4:-1,1],
                               data_train[0:-5,2],data_train[1:-4,2],data_train[2:-3,2],data_train[3:-2,2],data_train[4:-1,2],
                               data_train[0:-5,3],data_train[1:-4,3],data_train[2:-3,3],data_train[3:-2,3],data_train[4:-1,3],
                               data_train[0:-5,4],data_train[1:-4,4],data_train[2:-3,4],data_train[3:-2,4],data_train[4:-1,4],
                               data_train[0:-5,5],data_train[1:-4,5],data_train[2:-3,5],data_train[3:-2,5],data_train[4:-1,5]
                               ),axis=0).reshape(30,len(data_train[3:-2,3])).T
        _Ytr = data_train[5:,0,None]
        _Xts = np.concatenate((data_test[0:-5,0],data_test[1:-4,0],data_test[2:-3,0],data_test[3:-2,0],data_test[4:-1,0],
                               data_test[0:-5,1],data_test[1:-4,1],data_test[2:-3,1],data_test[3:-2,1],data_test[4:-1,1],
                               data_test[0:-5,2],data_test[1:-4,2],data_test[2:-3,2],data_test[3:-2,2],data_test[4:-1,2],
                               data_test[0:-5,3],data_test[1:-4,3],data_test[2:-3,3],data_test[3:-2,3],data_test[4:-1,3],
                               data_test[0:-5,4],data_test[1:-4,4],data_test[2:-3,4],data_test[3:-2,4],data_test[4:-1,4],
                               data_test[0:-5,5],data_test[1:-4,5],data_test[2:-3,5],data_test[3:-2,5],data_test[4:-1,5]
                               ),axis=0).reshape(30,len(data_test[3:-2,3])).T
        _Yts = data_test[5:,0,None]
        #remove redundant regressors with 'nan' values
        _Xtr,_Ytr=remove_nan(_Xtr,_Ytr)
        _Xts,_Yts=remove_nan(_Xts,_Yts)
        #add noise to output values
        _Ytr+= np.random.randn(_Ytr.shape[0])[:,None]* 0.01
        _Yts+= np.random.randn(_Yts.shape[0])[:,None]* 0.01
        return _Xtr,_Ytr,_Xts,_Yts
    
def split_data(data_type,X,Y,N,Ntr):
    """
    splits data into training and test
    """
    if data_type=='silverbox':
        """
        do nothing because everything is already done
        """
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
        """
        do nothing because everything is already done
        """
        
    return Xtr,Ytr,Xts,Yts

def rescale(X):
    """
    removes the average and sets standard deviation to 1
    """
    Xmean = np.nanmean(X,axis=0)
    X = np.subtract(X,Xmean)
    Xstd = np.nanstd(X,axis=0)
    X = np.divide(X,Xstd)  
    return X