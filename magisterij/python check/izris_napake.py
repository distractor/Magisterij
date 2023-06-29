# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:15:49 2017

@author: Mitja Jancic
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

part=np.arange(1999,2050,1,dtype='int')
part=np.arange(0,17295,1,dtype='int')


var=data[:,1]
pred=data[:,0]
Yts=data[:,2]


abs_error = np.asarray([2*sd for sd in np.sqrt(var)])
tOrig=np.arange(1,len(Yts)+1,1)
temp=np.asarray([y_i-sys_i for y_i,sys_i in zip(pred,Yts)])


plt.figure(figsize=(7,2.5))
plt.fill_between(tOrig[part],np.transpose(abs_error[part])[0], facecolor='#000000', alpha=0.3, interpolate=True, label='$2\sigma $')
plt.plot(tOrig[part],np.abs(temp[part]), 'k-', linewidth=0.1, label= '$|e|$')
plt.grid()
plt.xlabel('k')
plt.ylabel('e')
plt.title('Globoki GP')
plt.legend(shadow=True,loc=1) 
plt.xlim([tOrig[part][0],tOrig[part][-1]])
plt.tight_layout()
plt.show()
