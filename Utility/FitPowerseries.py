import numpy as np
import math
import os.path 
import sys
import csv
import matplotlib as mpb
import matplotlib.pyplot as plt
import dill
from math import sqrt
from AnalyzeMicrodisk import *
from uPL import *
from PlotPdep import *

def FitSpectrumDoublet(tag,j,x,y,e0,llo,lhi,l1,l2,DEBUG=False):
	
	ybg 	= GetBackground(x,y)
	if not os.path.exists("Fits/%s" % (tag)):
		os.makedirs("Fits/%s" % (tag))
	#If user specifies STARTI, don't start from 0
	STARTI 			= 0
	i 		= np.argmin(np.abs(x-e0))
	print "\tFitting doublet"
	[imn,imx,lor1,lor2] 	= FitDoublet(x,y,i,IDX=75,DEBUG=False,llo=lhi,lhi=llo,l1=l2,l2=l1)
	pbg 					= lor1.GetBackgroundFunction()
	xi 						= x[imn:(imx+1)]
	yi 						= y[imn:(imx+1)]
	l1 						= lor1.GetLambda0()
	fwhm1 					= np.sqrt(lor1.GetFWHM()**2)#-0.05**2)
	q1 						= l1 / fwhm1
	l2 						= lor2.GetLambda0()
	fwhm2 					= np.sqrt(lor2.GetFWHM()**2)#-0.05**2)
	q2 						= l2 / fwhm2			
	print "i: %d lam1: %0.2f Q1: %d lam2: %0.2f Q2: %d" % (j+STARTI, l1, q1, l2, q2)
	plt.plot(xi,yi,'k.')
	plt.plot(xi,pbg(xi),'b-',lw=1)
	plt.plot(xi,lor1.GetFitFunction()(xi),'r--',lw=1)		
	plt.plot(xi,lor2.GetFitFunction()(xi),'r--',lw=1)	
	plt.plot(xi,lor1.GetFitFunction()(xi)+lor2.GetFitFunction()(xi)-pbg(xi),'g-',lw=1.5) #Because the background is counted twice			
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts ()')
	plt.grid()
	plt.show(block=False)	
	accept 	= 1#int(raw_input("Accept (1 - Yes, 2 - No) -> "))
	if (accept == 1):				
		if not DEBUG:
			plt.savefig("Fits/%s/%s-%d.png" % (tag, tag, j+STARTI), bbox_inches='tight')						
		plt.close()	
		return [[lor1.GetLambda0(),lor1.GetFWHM(),lor1.GetA()+lor1.GetFitFunction()(lor1.GetLambda0())],[lor2.GetLambda0(),lor2.GetFWHM(),lor2.GetA()+\
		+lor2.GetFitFunction()(lor2.GetLambda0())]]		
	else:
		return [None, None]


def FitPowerseriesDoublet(tag,ts,l0,llo,lhi,l1,l2,STARTI=0,PMIN=None):
	ti 		= np.sort(ts.GetPowers())[-1::-1]
	x 		= ts.GetWavelength()
	tvec 	= []
	lvec 	= []
	qvec 	= []
	avec 	= []
	l2vec 	= []
	q2vec 	= []
	a2vec 	= []	
	#If user specifies STARTI, don't start from 0
	if STARTI > 0 and os.path.exists('%s.pkl'% (tag)):
		print "Loading from file!"
		ii 				= np.argmin(np.abs(ti-STARTI)) + 1
		if ii >= ti.shape[0]:
			print "Invalid start index"
			return
		ti 				= ti[ii:]	
		ddict 			= dill.load(open('%s.pkl' % (tag),'rb'))	
		tvec 			= ddict['t'].tolist()
		lvec 			= ddict['l'].tolist()
		qvec 			= ddict['q'].tolist()
		avec 			= ddict['a'].tolist()
		l2vec 			= ddict['l'].tolist()
		q2vec 			= ddict['q'].tolist()
		a2vec 			= ddict['a'].tolist()		
	for i in np.arange(0,ti.shape[0]):
		y 		= ts.GetSingleSpectrum(ti[i])
		[f1,f2] 	= FitSpectrum(tag,ti[i],x,y,l0)
		if f1 is not None and f2 is not None:
			tvec.append(ti[i])			
			lvec.append(f1[0])
			qvec.append(f1[1])
			avec.append(f1[2])
			l2vec.append(f2[0])
			q2vec.append(f2[1])
			a2vec.append(f2[2])			
			tarr 	= np.array(tvec)
			larr 	= np.array(lvec)
			qarr 	= np.array(qvec)
			aarr 	= np.array(avec)
			l2arr 	= np.array(l2vec)
			q2arr 	= np.array(q2vec)
			a2arr 	= np.array(a2vec)			
			dill.dump({'p':tarr,'l1':larr,'q1':qarr,'a1':aarr,'l2':l2arr,'q2':q2arr,'a2':a2arr},open('%s.pkl' % (tag),'wb'))
		if PMIN is not None:
			if ti[i] <= PMIN:
				return
			
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[18,18*2.0/(sqrt(5)+1)]}	
	matplotlib.rcParams.update(params)
	
	#Plot the destruction
	f1 	=	 "U:/Spectroscopy/QOLab/20170725-A3572-1-SiO2-5K-Anneal/wf_pwr_mapscan1.csv"
	
	ps0 = PowerSweep(f1,100,1800,"Destroy")
	PlotWaterfall(ps0)
	plt.gca().set_yscale('log')
	plt.xlim([400,402])
	plt.ylim([1000,3e4])
	plt.savefig('20170725-A3572-1-SiO2-5K-Anneal.pdf',bbox_inches='tight')	
	plt.savefig('20170725-A3572-1-SiO2-5K-Anneal.png',bbox_inches='tight')		
	plt.show()
	plt.clf()
	sys.exit()
	FitPowerseries("20170725_loc1-test",ps0,401.)	


