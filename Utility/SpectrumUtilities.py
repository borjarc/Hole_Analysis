from uPL import *
import matplotlib as mpb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import dill

def PlotIntensity(pwr,aArr,lamArr,wl,tint,p325,clr,CORRPOW=False,marks=None,label=None,IS405=False,NORM=False):
	#For power series from different measurements on same plot

	if marks is None:
		marks 	= ['o--']
	for i in [FindIndex(wl,lamArr)]:#range(aArr.shape[1]):
		if np.any(np.nonzero(aArr[:,i])):
			scale 	= np.ones(pwr.shape) / tint
			ii 		= np.nonzero(aArr[:,i])		
			if NORM:
				maxA 	= np.max(scale[ii]*aArr[ii,i])
			else:
				maxA 	= 1.
			if CORRPOW:
				scaleX 		= 2./3.
				scale325 	= 1.
			else:
				scaleX 		= 1.
				scale325 	= 2./3.			
			if IS405:
				plt.loglog( 1450.*scaleX, np.mean(scale[ii]*aArr[ii,i][0,:]), marks[0],lw=1.5,color=clr,ms=8)			
			else:
				if label is None:
					plt.loglog( (pwr[ii]-p325*scale325)*scaleX, scale[ii]*aArr[ii,i][0,:]/maxA, marks[0], lw=1.5, color=clr, ms=8)
				else:
					plt.loglog( (pwr[ii]-p325*scale325)*scaleX, scale[ii]*aArr[ii,i][0,:]/maxA, marks[0], lw=1.5, color=clr, ms=8,label=label)
			
def PlotLambda(pwr,lamArr,wl,p325,clr,CORRPOW=False,yerr=0.15,IS405=False):
	#For power series from different measurements on same plot
	marks 	= ['o--']
	for i in [FindIndex(wl,lamArr)]:#range(lamArr.shape[1]):
		if np.any(np.nonzero(lamArr[:,i])):
			ii 		= np.nonzero(lamArr[:,i])
			pwri 	= pwr[ii]
			lami 	= lamArr[ii,i][0,:]
			lamMn 	= lami[np.argmin(pwri)]
			ei 		= 1000.*(1240./lami - 1240./wl)
			eMn 	= 1240./lamMn
			print eMn
			if CORRPOW:
				scaleX 		= 2./3.
				scale325 	= 1.
			else:
				scaleX 		= 1.
				scale325 	= 2./3.
			if IS405:
				plt.errorbar( 1450.*scaleX, np.mean(ei), yerr=yerr, fmt=marks[0],lw=1.5,color=clr,ms=8)
			else:
				plt.errorbar( (pwri-p325*scale325)*scaleX, (ei), yerr=yerr*np.ones(pwri.shape[0]), fmt=marks[0], lw=1.5, color=clr, ms=8 )
			
def PlotFWHM(pwr,lamArr,fwhmArr,wl,p325,clr,CORRPOW=False,yerr=150.,IS405=False):
	#For power series from different measurements on same plot
	marks 	= ['-']
	for i in [FindIndex(wl,lamArr)]:#in range(fwhmArr.shape[1]):
		if np.any(np.nonzero(fwhmArr[:,i])):
			ii 		= np.nonzero(fwhmArr[:,i])
			pwri 	= pwr[ii]
			fwhmi 	= fwhmArr[ii,i][0,:]
			fwhmMn 	= fwhmi[np.argmin(pwri)]
			lami 	= lamArr[ii,i][0,:]		
			lamMn 	= lami[np.argmin(pwri)]			
			ei 		= 1e6*1240.*fwhmi/np.power(lami,2)
			emn 	= 1e6*1240.*fwhmMn/np.power(lamMn,2)
			if CORRPOW:
				scaleX 		= 2./3.
				scale325 	= 1.
			else:
				scaleX 		= 1.
				scale325 	= 2./3.			
			if IS405:
				plt.errorbar( 1450.*scaleX, np.mean(ei), yerr=yerr, fmt=marks[0],lw=1.5,color=clr,ms=8)
			else:
				plt.errorbar( (pwri-p325*scale325)*scaleX, ei, yerr=yerr*np.ones(pwri.shape[0]), fmt=marks[0], lw=1.5, color=clr, ms=8 )			

def FindIndex(wl,lamArr):
	avs 	= np.zeros(lamArr.shape[0])
	for j in range(lamArr.shape[1]):
		lamj 	= lamArr[:,j]
		avs[j] 	= np.mean(lamj[lamj>0])
	return np.argmin(np.abs( wl - avs ))

def ProcessPowerSeries(pobj,LMIN,LMAX,DK=1040.):
	'''
	Work with a 150 lp/mm power series.  Get the peak position and area as as function of power
	'''
	pwr 	= pobj.GetPowers()
	pwr 	= np.sort(pwr)
	eav 	= np.zeros(pwr.shape[0])
	iint 	= np.zeros(pwr.shape[0])
	for j in range(pwr.shape[0]):
		up1 			= pobj.GetSingleUPL(pwr[j])
		eav[j] 			= IntegralAverageWL(up1,LMIN,LMAX,DK=DK)
		iint[j] 		= IntegrateSpectrum(up1,LMIN,LMAX,DK=DK)
	return [pwr,eav,iint]

def IntegrateSpectrum(lsobj,LMIN,LMAX,DK=1040.):
	'''
	Get the average wavelength by integrating the spectrum from 
	'''
	wl 			= lsobj.GetWavelength()
	cts 		= lsobj.GetCounts()
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	#plt.plot(wl,cts-DK)
	#plt.show()
	return np.abs(np.trapz(cts[ii]-DK,x=wl[ii]))

def IntegralAverageWL(lsobj,LMIN,LMAX,DK=1040.):
	'''
	Get the average wavelength by integrating the spectrum from  LMIN to LMAX
	'''
	wl 			= lsobj.GetWavelength()
	cts 		= lsobj.GetCounts()
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	'''
	cts 		= cts[ii]
	wl 			= wl[ii]
	ipk     	= np.argmax(cts)
	jj 			= np.logical_and(wl>=(wl[ipk]-3.5),wl<=(wl[ipk]+3.5))
	p 			= np.polyfit(1240./wl[jj],cts[jj],2)
	return -1240./(p[1]*0.5/p[0])#
	'''
	return np.trapz(wl[ii]*(cts[ii]-DK),x=wl[ii])/np.trapz(cts[ii]-DK,x=wl[ii])
	
def GetMaxWL(lsobj,LMIN,LMAX,DK=1040.,WINDOW=3.):
	'''
	Get the average wavelength by integrating the spectrum from  LMIN to LMAX
	'''
	wl 			= lsobj.GetWavelength()
	cts 		= lsobj.GetCounts()
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	cts 		= medfilt(cts,5)
	cts 		= cts[ii] - DK
	wl 			= wl[ii]
	ipk     	= np.argmax(cts)
	jj 			= np.logical_and(wl>=(wl[ipk]-WINDOW),wl<=(wl[ipk]+WINDOW))
	cts[jj] 	= cts[jj] - np.min(cts[jj])+ 0.0001
	p 			= np.polyfit(1240./wl[jj],np.log(cts[jj]),2)
	return -1240./(p[1]*0.5/p[0])#
	
def GetMyFWHM(lsobj,LMIN,LMAX,DK=970.,WINDOW=7.):
	'''
	Get the average wavelength by integrating the spectrum from  LMIN to LMAX
	'''
	wl 			= lsobj.GetWavelength()
	cts 		= lsobj.GetCounts()
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	cts 		= medfilt(cts,5)
	cts 		= cts[ii] - DK
	wl 			= wl[ii]
	ipk     	= np.argmax(cts)
	jj 			= np.logical_and(wl>=(wl[ipk]-WINDOW),wl<=(wl[ipk]+WINDOW))
	cts[jj] 	= cts[jj] - np.min(cts[jj])+ 0.0001
	p 			= np.polyfit(1240./wl[jj],np.log(cts[jj]),2)	

	#DK 			= 974.
	emax 		= 1240. / GetMaxWL(lsobj,LMIN,LMAX,DK,WINDOW)
	en 			= 1240./lsobj.GetWavelength()
	wl 			= 1240./en
	cts 		= lsobj.GetCounts() - DK 
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	cts 		= medfilt(cts,5)
	cts 		= cts[ii] - np.min(cts[ii])+ 0.0001
	en 			= en[ii]
	ii 			= np.argsort(en)
	en 			= en[ii]
	cts 		= cts[ii]
	imx 		= np.argmin(np.abs(en-emax))
	cmax 		= cts[imx]
	off 		= np.exp(np.polyval(p,emax))	
	i 			= imx - 1
	while ( (cts[i] - off) > 0.5*(cmax-off)) and (i > 1):
		i 		= i - 1
	if i == 1:
		return 0 
	else:
		return 4*np.abs(0.5*(en[i]+en[i-1])-emax)
	'''
	wl 			= lsobj.GetWavelength()
	cts 		= lsobj.GetCounts()
	ii 			= np.logical_and(wl>=LMIN,wl<=LMAX)
	cts 		= medfilt(cts,5)
	cts 		= cts[ii] - DK
	wl 			= wl[ii]
	ipk     	= np.argmax(cts)
	jj 			= np.logical_and(wl>=(wl[ipk]-WINDOW),wl<=(wl[ipk]+WINDOW))
	cts[jj] 	= cts[jj] - np.min(cts[jj])+ 0.0001
	p 			= np.polyfit(1240./wl[jj],np.log(cts[jj]),2)
	return 4.*np.sqrt(4*np.log(2)/np.abs(2*p[0]))
	'''

if __name__=="__main__":
	pass