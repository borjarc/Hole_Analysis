# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib as mpb
import sys

def CalcOneTau(tph1,tph2,tau,dtau):
    tph2i   = tph2 + tau
    N       = 0
    for t2 in tph2i:
        N   = N + np.nonzero(np.abs(t2-tph1) <= dtau)[0].shape[0]
    return N
    

def CalculateG2(tph1,tph2,taumax,tag="",dtau=4):
    '''
    Calculate the g(2) function for the time series tph1 and tph2 in picoseconds
    taumax is the maximum tau value in picoseconds
    '''
    Ntau    = int(math.floor(2*taumax/dtau))
    tau     = np.arange(0,Ntau+1)*dtau-taumax
    g2num   = np.zeros(tau.shape)
    g2den   = tph1.shape[0]*tph2.shape[0]
    i       = 0
    tph1    = np.sort(tph1)
    tph2    = np.sort(tph2)
    #f       = plt.plot([],[],'k-',lw=2)   
    #plt.xlabel('$\tau$ [ns]')
    #plt.ylabel('$g^{2}$')
    ##plt.grid()
    #plt.show(block=False)   
    i1      = 0 
    i2l     = 0
    i2h     = 1        
    for i1 in range(tph1.shape[0]):
        
        if ( i1 % 10000 == 1):
            print "%0.1f percent complete" % (100.0*i/tph1.shape[0])
            plt.plot(tau/1000.,g2num,'k-')
            plt.xlim([-200,200])
            plt.xlabel('$\\tau$ [ns]')
            plt.ylabel('$g^{(2)}(\\tau)$')
            plt.grid()
            plt.savefig('%s-g2.png' % (tag))
            plt.close()
            #f.set_xdata(tau)
            #f.set_ydata(g2num)
            #plt.draw()
        
        while ( (tph2[i2h] - tph1[i1]) < taumax):
            if i2h == tph2.shape[0] - 1:
                break
            i2h     = i2h + 1
            
        while (tph1[i1] - tph2[i2l] > taumax):
            if i2l == i2h:
                break
            i2l     = i2l + 1
        if (tph2[i2h] - tph1[i1]) > taumax:
            continue
        for i2 in range(i2l,i2h+1):
            taui    = tph2[i2] - tph1[i1]
            g2i     = int(math.floor(np.round((taui+taumax)/dtau)))
            g2num[g2i]  = g2num[g2i] + 1

    return [tau, g2num / g2den]
            
if __name__=="__main__":
    filen   = sys.argv[1]
    params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 18,
    'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(math.sqrt(5)+1)]}	
    mpb.rcParams.update(params)	 
    dat=loadmat('U:/Spectroscopy/QOLab/20161011-A3405C-g2/%s' % (filen))
    tph1=np.array(dat['tph1'][0])
    tph2=np.array(dat['tph2'][0])
    [tau,g2] = CalculateG2(tph1,tph2,5e5,filen)
    plt.plot(tau/1000.,g2,'k-')
    plt.xlim([-200,200])
    savemat('2mW_g2.mat',{'tau':tau,'g2':g2})
    plt.show()