# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 14:59:50 2017

@author: Mitja Jancic
"""

import pandas as pd
import plotgp as pgp
import numpy as np

filepath='C:/Users/mitja/Documents/magisterij/python check/results/years/'
#f = 'results_sinteticni_100ind_noise005.txt'
f = 'results_rand_bestDGP.txt'
data = pd.read_csv(filepath+f, header=0 ,delimiter='\t').values


t=np.arange(1,len(data[:,0])+1,1)

#pgp.plotgp(2,t,data[:,4],data[:,2],np.sqrt(data[:,3]),partly=True)
#pgp.plotgp(3,t,data[:,4],data[:,0],np.sqrt(data[:,1]),partly=True)
#
pgp.plotgp(3,t,data[:,2],data[:,0],np.sqrt(data[:,1]),partly=True)
