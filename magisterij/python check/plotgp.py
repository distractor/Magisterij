# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 12:10:03 2017

@author: Mitja Jancic
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def plotgp(i, ti, sys ,y ,std, partly=True):
    """
    t - time vector, x axis
    sys - target output
    y - predicted output
    std - predicted standard deviation
    """
    tOrig=ti.copy()
    ix_plot = np.arange(0,len(tOrig),1).tolist();   
    
    mean_up = np.asarray([t+2*sd for t,sd in zip(y[ix_plot],std[ix_plot])])
    mean_down = np.asarray([t-2*sd for t,sd in zip(y[ix_plot],std[ix_plot])])
    
    if partly==False:
        #if i==0 use current axis (figure) 
        if (i!=0):
            plt.figure(i)
        
        # upper part of figure: y 
        gs1 = gridspec.GridSpec(3, 3)
        ax1 = plt.subplot(gs1[:-1, :])
        ax1.plot(tOrig,y, 'k-', linewidth=1,label='$\mu$')
        ax1.plot(tOrig,sys, 'k--', linewidth=2, label='sistem')
        ax1.fill_between(tOrig,mean_up,mean_down, facecolor='#000000', alpha=0.3, interpolate=True, label='$\mu \pm 2\sigma$')
        ax1.grid()
        ax1.set_xlabel('k')
        ax1.set_xlim([tOrig[0],tOrig[-1]])
        ax1.set_ylabel('y') 
        if i==2:
            ax1.set_title('GP-LVM')
        else:
            ax1.set_title('Globoki GP')
        ax1.legend(shadow=True,loc=1) 
        ax2 = plt.subplot(gs1[-1, :])
        abs_error = np.asarray([2*sd for sd in std[ix_plot]])
        ax2.fill_between(tOrig,abs_error,0, facecolor='#000000', alpha=0.3, interpolate=True, label='$2\sigma $')
        temp=np.asarray([y_i-sys_i for y_i,sys_i in zip(y[ix_plot],sys[ix_plot])])
        ax2.plot(tOrig,np.abs(temp), 'k-', linewidth=1, label= '$|e|$')
        ax2.legend(shadow=True,loc=1) 
        ax2.grid()
        ax2.set_xlabel('k')
        ax2.set_xlim([tOrig[0],tOrig[-1]])
        ax2.set_ylabel('e') 
        plt.tight_layout()
        plt.show()
    else:
        part=np.arange(7999,8200,1,dtype='int')
        #if i==0 use current axis (figure) 
        if (i!=0):
            plt.figure(i)
        
        # upper part of figure: y 
        gs1 = gridspec.GridSpec(3, 3)
        ax1 = plt.subplot(gs1[:-1, :])
        ax1.plot(tOrig[part],y[part], 'k-', linewidth=1,label='$\mu$')
        ax1.plot(tOrig[part],sys[part], 'k--', linewidth=2, label='sistem')
        ax1.fill_between(tOrig[part],mean_up[part],mean_down[part], facecolor='#000000', alpha=0.3, interpolate=True, label='$\mu \pm 2\sigma$')
        ax1.grid()
        ax1.set_xlabel('k')
        ax1.set_xlim([tOrig[part][0],tOrig[part][-1]])
        ax1.set_ylabel('y') 
        if i==2:
            ax1.set_title('GP-LVM')
        else:
            ax1.set_title('Globoki GP')
        ax1.legend(shadow=True,loc=1) 
        ax2 = plt.subplot(gs1[-1, :])
        abs_error = np.asarray([2*sd for sd in std[ix_plot]])
        ax2.fill_between(tOrig[part],abs_error[part],0, facecolor='#000000', alpha=0.3, interpolate=True, label='$2\sigma $')
        temp=np.asarray([y_i-sys_i for y_i,sys_i in zip(y[ix_plot],sys[ix_plot])])
        ax2.plot(tOrig[part],np.abs(temp[part]), 'k-', linewidth=1, label= '$|e|$')
        ax2.legend(shadow=True,loc=1) 
        ax2.grid()
        ax2.set_xlabel('k')
        ax2.set_xlim([tOrig[part][0],tOrig[part][-1]])
        ax2.set_ylabel('e') 
        plt.tight_layout()
        plt.show()
