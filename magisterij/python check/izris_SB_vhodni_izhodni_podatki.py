# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:03:12 2017

@author: Mitja Jancic
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd

silverbox_data = scipy.io.loadmat('C:/Users/mitja/Documents/magisterij/sinteticni/sb.mat')
#training and test data
sb_train = silverbox_data['data_train']
sb_test = silverbox_data['data_test']

fig = plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(sb_train[:300,0],'k-o',label='u(k)',markersize=3)
plt.xlabel('k')
plt.grid()
plt.legend(shadow=True)
plt.xlim(0,300)
plt.subplot(2,1,2)
plt.plot(sb_train[:300,1],'k-o',label='y(k)',markersize=3)
plt.xlabel('k')
plt.legend(shadow=True)
plt.xlim(0,300)
plt.grid()
plt.show()
plt.tight_layout()
fig.savefig('C:/Users/mitja/Documents/magisterij/latex/sinteticni/vhod_izhod.pdf',fig)