import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from scipy.stats import linregress
import cPickle as pickle

def MakeLamQDoubleAxisPlot(hBias, lamB, QB, lamR, QR):

	fig, ax1 = plt.subplots()
	fig.set_size_inches(1.1*8,6*1.1)
	ax1.set_xlabel('Hole radius bias [nm]')
	# Make the y-axis label and tick labels match the line color.
	ax1.set_ylabel('Q', color='k')
	for tl in ax1.get_yticklabels():
		tl.set_color('k')
	ax2 = ax1.twinx()	
	ax2.set_ylabel('Wavelength [nm]', color='g')
	for tl in ax2.get_yticklabels():
		tl.set_color('g')
	lBmu 		= np.array(map(lambda x: np.mean(x), lamB))
	lBstd		= np.array(map(lambda x: np.std(x), lamB)) 
	QBmu 		= np.array(map(lambda x: np.mean(x), QB))
	QBstd		= np.array(map(lambda x: np.std(x), QB)) 
	lRmu 		= np.array(map(lambda x: np.mean(x), lamR))
	lRstd		= np.array(map(lambda x: np.std(x), lamR)) 
	QRmu 		= np.array(map(lambda x: np.mean(x), QR))
	QRstd		= np.array(map(lambda x: np.std(x), QR))	
	
	ax2.errorbar(hBias,lBmu,fmt='go-',yerr=lBstd,ms=8,lw=2,mfc='b')
	ax2.errorbar(hBias,lRmu,fmt='gs-',yerr=lRstd,ms=8,lw=2,mfc='r')
	ax1.errorbar(hBias,QBmu,fmt='ko-',yerr=QBstd,ms=8,lw=2,mfc='b')
	ax1.errorbar(hBias,QRmu,fmt='ks-',yerr=QRstd,ms=8,lw=2,mfc='r')
	ax2.set_xlim([np.min(hBias)-1, np.max(hBias)+1])
	ax1.set_xlim([np.min(hBias)-1, np.max(hBias)+1])	
	ax1.set_ylim([0,8000])
	
def MakeAQDoubleAxisPlot(hBias, AB, QB, AR, QR, DBL=True):
	fig, ax1 = plt.subplots()
	fig.set_size_inches(1.1*8,6*1.1)
	ax1.set_xlabel('Peak-to-peak grating amplitude [nm]')
	# Make the y-axis label and tick labels match the line color.
	ax1.set_ylabel('Relative intensity', color='b')
	for tl in ax1.get_yticklabels():
		tl.set_color('b')
	ax2 = ax1.twinx()	
	ax2.set_ylabel('Q', color='r')
	for tl in ax2.get_yticklabels():
		tl.set_color('r')
	lBmu 		= np.array(map(lambda x: np.mean(x), AB))
	lBstd		= np.array(map(lambda x: np.std(x), AB)) 
	QBmu 		= np.array(map(lambda x: np.mean(x), QB))
	QBstd		= np.array(map(lambda x: np.std(x), QB)) 
	lRmu 		= np.array(map(lambda x: np.mean(x), AR))
	lRstd		= np.array(map(lambda x: np.std(x), AR)) 
	QRmu 		= np.array(map(lambda x: np.mean(x), QR))
	QRstd		= np.array(map(lambda x: np.std(x), QR))	
	
	ax1.errorbar(hBias,lBmu/lBmu[0],fmt='bo:',yerr=lBstd/(2*lBmu[0]),ms=8,lw=1.5)
	if DBL:
		ax1.errorbar(hBias,lRmu/lRmu[0],fmt='bs:',yerr=lRstd/(2*lRmu[0]),ms=8,lw=1.5)
	ax2.errorbar(hBias,QBmu,fmt='ro:',yerr=QBstd,ms=8,lw=1.5)
	if DBL:
		ax2.errorbar(hBias,QRmu,fmt='rs:',yerr=QRstd,ms=8,lw=1.5)
	ax1.set_xlim([np.min(hBias)-1, np.max(hBias)+1])
	ax2.set_xlim([np.min(hBias)-1, np.max(hBias)+1])	
	ax2.set_ylim([0,8000])	
	
def PlotXY(xlst, ylst, clr='k', lab=None):
	for i in range(len(xlst)):
		if lab is not None and i == 0:
			plt.plot(xlst[i],ylst[i],'%s.'%(clr),ms=8, label=lab)
		else:
			plt.plot(xlst[i],ylst[i],'%s.'%(clr),ms=8)
			
def PlotXYSlog(xlst, ylst, clr='k', lab=None):
	for i in range(len(xlst)):
		if lab is not None and i == 0:
			plt.semilogy(xlst[i],ylst[i],'%s.'%(clr),ms=8, label=lab)
		else:
			plt.semilogy(xlst[i],ylst[i],'%s.'%(clr),ms=8)	
			
def PlotXYLamQ(a, lam, Q, lut, clr='k', lab=None, SEMIY=False, sig=1.0):
	j 	= 0
	l 	= np.array([])
	q 	= np.array([])
	for i in range(len(lam)):
		lami 		= np.array(lam[i])
		Qi			= np.array(Q[i])
		ai 			= np.array(a[i])
		fwhm 		= lami / Qi
		if len(lam[i]) == 0 or len(Q[i]) == 0:
			continue
		xerri		= lut.EstimateWL(fwhm,ai,sig)
		yerri 		= lut.EstimateFWHM(fwhm,ai,sig)
		Qerri 		= np.sqrt( np.power(xerri/fwhm,2) + np.power( lam[i]*yerri/np.power(fwhm,2), 2) )
		if lab is not None and j == 0:
			plt.errorbar(lam[i],Q[i],xerr=2*xerri,yerr=2*Qerri,fmt='.',color=clr,ms=6, label=lab)
			j 	= j + 1
		else:
			plt.errorbar(lam[i],Q[i],xerr=2*xerri,yerr=2*Qerri,fmt='.',color=clr,ms=6)
		l 	= np.hstack([l,lam[i]])
		q 	= np.hstack([q,Q[i]])
	if SEMIY:
		ax 	= plt.gca()
		ax.set_yscale('log')
	return [l,q]
		
def PlotXYLamA(a, lam, Q, lut, clr='k', lab=None, SEMIY=True, sig=1.0):
	j 	= 0
	for i in range(len(lam)):
		lami 		= np.array(lam[i])
		Qi			= np.array(Q[i])
		ai 			= np.array(a[i])
		fwhm 		= lami / Qi
		if len(lam[i]) == 0 or len(a[i]) == 0:
			continue
		xerri		= lut.EstimateWL(fwhm,ai,sig)
		yerri 		= lut.EstimateFWHM(fwhm,ai,sig)
		Qerri 		= np.sqrt( np.power(xerri/fwhm,2) + np.power( lam[i]*yerri/np.power(fwhm,2), 2) )		
		aerri 		= lut.EstimateA(fwhm,ai,sig)
		#yerri 		= np.sqrt( np.power(aerri/Q[i], 2) + np.power( a[i]*Qerri/np.power(Q[i],2), 2) )	
		yerri 		= np.sqrt( np.power(aerri*Q[i], 2) + np.power( Qerri*a[i], 2) )			
		if lab is not None and j == 0:
			plt.errorbar(lam[i],a[i]*Q[i],xerr=xerri,fmt='.',color=clr,ms=8, label=lab)
			j 	= j + 1
		else:
			plt.errorbar(lam[i],a[i]*Q[i],xerr=xerri,fmt='.',color=clr,ms=8)
	if SEMIY:
		ax 	= plt.gca()
		ax.set_yscale('log')		
		
def PlotXYDiameterQ(r, a, lam, Q, lut, clr='k', lab=None, SEMIY=False, sig=1.0):
	j 	= 0
	for i in range(len(lam)):
		lami 		= np.array(lam[i])
		Qi			= np.array(Q[i])
		ai 			= np.array(a[i])
		fwhm 		= lami / Qi
		if len(lam[i]) == 0 or len(Q[i]) == 0:
			continue
		xerri		= lut.EstimateWL(fwhm,ai,sig)
		yerri 		= lut.EstimateFWHM(fwhm,ai,sig)
		Qerri 		= np.sqrt( np.power(xerri/fwhm,2) + np.power( lam[i]*yerri/np.power(fwhm,2), 2) )
		if lab is not None and j == 0:
			plt.errorbar(2*r[i]*np.ones(Q[i].shape),Q[i],yerr=2*Qerri,fmt='.',color=clr,ms=8, label=lab)
			j 	= j + 1
		else:
			plt.errorbar(2*r[i]*np.ones(Q[i].shape),Q[i],yerr=2*Qerri,fmt='.',color=clr,ms=8)
	if SEMIY:
		ax 	= plt.gca()
		ax.set_yscale('log')		
		
def PlotXYQA(a, lam, Q, lut, clr='k', lab=None, SEMIY=False, sig=1.0):
	j 				= 0
	for i in range(len(lam)):
		lami 		= np.array(lam[i])
		Qi			= np.array(Q[i])
		ai 			= np.array(a[i])
		fwhm 		= lami / Qi
		if len(lam[i]) == 0 or len(Q[i]) == 0:
			continue
		xerri		= lut.EstimateWL(fwhm,ai,sig)
		yerri 		= lut.EstimateFWHM(fwhm,ai,sig)
		Qerri 		= np.sqrt( np.power(xerri/fwhm,2) + np.power( lam[i]*yerri/np.power(fwhm,2), 2) )	
		yerri 		= lut.EstimateA(fwhm,ai,sig)	
		xerri 		= Qerri	
		if lab is not None and j == 0:
			plt.errorbar(Qi,ai,xerr=xerri,yerr=yerri,fmt='.',color=clr,ms=8, label=lab)
			j 	= j + 1
		else:
			plt.errorbar(Qi,ai,xerr=xerri,yerr=yerri,fmt='.',color=clr,ms=8)
	if SEMIY:
		ax 	= plt.gca()
		ax.set_yscale('log')						

def MakeHistogram(dat, bin, off, Nbin=15, color='b', lab=None):
	flatdat 	= np.array([])
	for di in dat:
		flatdat 	= np.hstack([flatdat, di])
	dat 		= flatdat
	hist, _ 	= np.histogram(dat, bins=bin)
	width 		= 0.70 * (bin[1] - bin[0])
	ctr 		= 0.5*(bin[1:] + bin[:-1])
	if lab is not None:
		plt.bar(ctr+off*width, 100*hist/np.sum(hist), align='center', width=width, color=color, label=lab)	
	else:
		plt.bar(ctr+off*width, 100*hist/np.sum(hist), align='center', width=width, color=color)
		
def FlattenList(xlst):
	xlst 		= filter(lambda x: x.shape[0] > 0, xlst)
	x 		= np.array([])
	for i in range(len(xlst)):
		x 	= np.hstack([x, xlst[i]])
	return x

def TestRegression(xlst,ylst,LOGY=False):
	x 	= FlattenList(xlst)
	if LOGY:
		y 	= np.log(FlattenList(ylst))
	else:
		y 	= FlattenList(ylst)
	slope, intercept, r_value, p_value, std_err 	= linregress(x,y)
	return [slope,intercept,r_value,p_value]
	