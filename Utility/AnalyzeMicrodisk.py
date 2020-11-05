import numpy as np
import math
import os.path 
import os
import sys
import csv
import matplotlib as mpb
import matplotlib.pyplot as plt
from uPL import *
from scipy import signal
from numpy import fft
import scipy.interpolate as spip
import dill

def GetProblemWL():
	return np.array([410.7,430.0,449.16,468.17,486.76]) - 5.7

def RemoveProblemWL(x,ii,TOL=0.5):
	gluewl 	= GetProblemWL()
	dist 	= np.abs(np.outer(gluewl,np.ones(ii.shape[0])) - np.outer(np.ones(gluewl.shape[0]), x[ii]))
	badind 	= np.nonzero(dist < TOL)
	return np.setdiff1d(ii,ii[badind[1]])	
	
def trapz(x,y):
	y2 	= (x[1:] - x[:-1]) * 0.5 * (y[1:] + y[:-1])
	return np.hstack([y2[0],y2])

def FindPeaks(x,y,NSMTH=15,NWINDOW=200, STD = 0.005, QMAX=8000,DEBUG=True): #NSMTH=15, STD = 0.02, QMAX=4000 #NSMTH=10,STD=0.005,QMAX=5000.
	ysmth 			= np.convolve(y,np.ones(NSMTH),'same') / NSMTH
	g 				= np.gradient(ysmth)
	g2 				= np.gradient(g)
	g2norm 			= g2/np.max(np.abs(g2))

	#Find the minima
	ii 				= signal.argrelextrema(g2norm,np.less)[0]
	ii 				= ii[np.logical_and(ii > 5, ii < ysmth.shape[0]-5)]
	def FilterFunction(i):
		imn 		= i - NWINDOW/2
		if imn < 0:
			imn 	= 2
		imx 		= i + NWINDOW/2 
		if imx > ysmth.shape[0]:
			imx 	= ysmth.shape[0] - 3
		try:
			ystd 		= np.mean(np.sqrt(ysmth[(imn+1):(imx+1)]+4*ysmth[(imn):(imx)]+ysmth[(imn-1):(imx-1)]))
		except:
			ystd 		= 40.
		return (np.abs(g2[i]) >= STD*ystd and g2[i] < 0)
	ii 				= np.array(filter(FilterFunction, ii.tolist()))
	#Filter out peaks that are too close to one another
	dist			= np.abs( np.outer(x[ii],np.ones(ii.shape[0])) - np.outer(np.ones(ii.shape[0]),x[ii]) )
	#Set the diagonals to large values 
	for i in range(dist.shape[0]):
		dist[i,i] 		= 100.
	badind 			= np.nonzero(dist <= (np.min(x) / QMAX))
	lst 			= np.unique(np.max(np.vstack([badind[0],badind[1]]),0))
	ii 				= np.setdiff1d(ii,ii[lst])
	ii 				= np.sort(RemoveProblemWL(x,ii))
	
	if DEBUG:
		ybg 		= GetBackground(x,y,ROLL=-250)
		plt.plot(x,y,'k-',lw=0.5)	
		plt.plot(x[ii],y[ii],'bo',ms=4)
		plt.plot(x,ybg,'r-')
		plt.show()
	#Now, remove peaks that fall close to the glue window points
	return ii
	
def GetBackground(x,y,norm_pass=0.005,norm_stop=0.05,ROLL=-250):
	#Fit the background
	fitp			= spip.interp1d(x,y,kind='linear')
	xlin 			= np.linspace(np.min(x),np.max(x),x.shape[0])
	ylin 			= fitp(xlin)	
	#Apply a low-pass filter to get rid of peaks
	(b, a) 		= signal.butter(4,0.005,btype='low')
	ybg 		= signal.lfilter(b, a, ylin)
	ybgNorm 	= ybg - np.min(ybg)
	ybgNorm 	= ybgNorm / np.max(ybgNorm)
	def FitFxn(x):
		rll 	= int(np.round(x[0]))
		scl 	= x[1]
		off 	= x[2]
		return y - np.roll(scl*ybgNorm+off,rll)
	x0 		= [ROLL,np.max(ybg-np.min(ybg)),np.min(ybg)]
	xfit 	= spop.leastsq(FitFxn, x0)[0]	
	#print "Roll: %d %d" % (x0[0],xfit[0])
	#print "Scale: %0.1e %0.1e" % (x0[1],xfit[1])
	#print "Offset: %0.1e %0.1e" % (x0[2],xfit[2])
	return np.roll(ybgNorm*xfit[1]+xfit[2],int(np.round(xfit[0])))

def FitLorentzian4(lam, ctrt, a0, lam0, fwhm0, pbg):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	a0: 	Initial guess on peak height
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	p_bgd:  Background level polynomial fit
	Returns:
	Lorentzian function fit object for best fit
	'''
	ctrt 	= ctrt - np.polyval(pbg, lam)
	#Pick the initial values
	x0 			= np.array([a0,lam0,fwhm0,0])	
	def FitFxn(x):
		A 			= np.abs(x[0])
		x0 			= x[1]
		gamma 		= x[2] / 2.0
		C 			= x[3]
		return (ctrt - A / (1.0 + np.power( (lam-x0)/gamma, 2) ) - C)
	#Do the fit
	xfit 	= spop.leastsq(FitFxn, x0)[0]
	pbg[-1] 	= pbg[-1] + xfit[3]
	bgfxn 	= lambda x: np.polyval(pbg,x)
	return Lorentzian(np.abs(xfit[0]), xfit[1], np.abs(xfit[2]), bgfxn)		
	
def FindInitialParameters(x,y,gbg,i0,TOL=0.2,NSMTH=5,DEBUG=False):
	'''
	Crawl across the peak starting at position i.  The bounds are where the signal stops decreasing
	'''
	ysmth 	 = np.convolve(y,np.ones(NSMTH),'same') / NSMTH
	y 		= ysmth
	#Crawl to the nearest local maximum
	ymx 	= y[i0]
	imxn 	= i0 - 1
	if imxn == 0:
		imxp 	= 1
	else:
		imxp 	= i0
	while imxn > 0 and y[imxn] > y[imxn+1]:
		imxn 	= imxn - 1
	while imxp < y.shape[0] and y[imxp] > y[imxp-1]:
		imxp 	= imxp + 1
	if y[imxp] > y[imxn]:
		imax 	= imxp - 1
	else:
		imax 	= imxn + 1

	lam0 		= x[imax]
	#Find the left-hand side of the peak
	ilo 		= imax - 1 
	while ilo > 1 and (y[ilo] < y[ilo+1] or y[ilo-1] < y[ilo]) :
		ilo 	= ilo - 1 
	if ilo < 1:
		ilo 	= 1
	
	#Find the right-hand side of the peak
	ihi 		= imax + 1 
	while ihi < x.shape[0]-1 and (y[ihi] < y[ihi-1] or y[ihi+1] < y[ihi]):
		ihi 	= ihi + 1 	
	if ihi >= x.shape[0]:
		ihi 	= x.shape[0]-1

	xbg 		= np.hstack([x[(ilo-2):ilo],x[ihi:(ihi+2)]])
	ybg 		= np.hstack([y[(ilo-2):ilo],y[ihi:(ihi+2)]])	
	pbg 		= np.polyfit(xbg, ybg, 1)
	a0 			= y[imax] - np.polyval(pbg,x[imax])	

	#Find the FWHM by crawling and praying
	cutM 	= 0.5*(y[ilo]+y[imax])
	cutP 	= 0.5*(y[ihi]+y[imax])
	ip 		= imax+1
	im 		= imax-1
	while im > 0 and y[im] > cutM:
		im 	= im - 1 
	if im < 0:
		im 	= 0
	#Find the right-hand side of the peak
	while ip < x.shape[0]-1 and y[ip] > cutP:
		ip 	= ip + 1 	
	if ip >= x.shape[0]:
		ip 	= x.shape[0] - 1	
	fwhm0 	= x[ip] - x[im]
	if DEBUG:
		xi 		= x[(imax-50):(imax+50)]
		yi 		= y[(imax-50):(imax+50)]
		plt.plot(xi,yi,'k.')
		plt.plot(xi,np.polyval(pbg,xi),'r-')
		plt.plot(x[imax],y[imax],'y^',ms=10)
		plt.plot(x[i0],y[i0],'bs',ms=10)
		plt.plot(x[ilo],y[ilo],'g^',ms=10)
		plt.plot(x[ihi],y[ihi],'g^',ms=10)		
		plt.show()
	
	return [ilo,ihi,a0,lam0,fwhm0,pbg]	
	
def GetMode(x,y,i,IDX=50):
	xi 		= x[(i-IDX):(i+IDX)]
	yi 		= y[(i-IDX):(i+IDX)]
	plt.plot(xi,yi,'k.-')
	plt.plot(x[i],y[i],'ro',ms=8)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts (arb)')
	plt.show(block=False)	
	mode 	= int(raw_input("Enter mode (1 singlet, 2 doublet, 3 ignore) -> "))
	plt.close()
	return mode
	
def FitDoublet(x,y,i,IDX=75,DEBUG=True,llo=None,lhi=None,l1=None,l2=None,PGBOVERRIDE=False):
	#Plot the data so we can take user input
	xi 		= x[(i-IDX):(i+IDX)]
	yi 		= y[(i-IDX):(i+IDX)]
	plt.plot(xi,yi,'k.-')
	plt.plot(x[i],y[i],'ro',ms=8)
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts (arb)')
	plt.show(block=False)
	
	if llo is None:
		llo 	= float(raw_input("\tEnter lower window (nm) -> "))
	imn 	= np.argmin(np.abs(x - llo))
	if lhi is None:
		lhi 	= float(raw_input("\tEnter upper window (nm) -> "))	
	imx 	= np.argmin(np.abs(x - lhi))

	#Fit the background
	xbg 		= np.hstack([x[(imn-2):imn],x[imx:(imx+2)]])
	ybg 		= np.hstack([y[(imn-2):imn],y[imx:(imx+2)]])	
	pbg 		= np.polyfit(xbg, ybg, 1)
	if PGBOVERRIDE:
		pbg 	= np.array([np.min(ybg)])
	#Estimate the inital peak heights
	if l1 is None:
		l1 		= float(raw_input("\tEnter peak 1 pos (nm) -> "))
	i1 		= np.argmin(np.abs(x-l1))
	if l2 is None:
		l2 		= float(raw_input("\tEnter peak 2 pos (nm) -> "))
	i2 		= np.argmin(np.abs(x-l2))	
	a1 		= y[i1] - np.polyval(pbg,x[i1])	
	a2 		= y[i2] - np.polyval(pbg,x[i2])
	
	#Estimate the FWHMs
	if1 	= i1 - 1
	while y[if1] >= 0.5*(y[imn]+y[i1]) and if1 > 1:
		if1 	= if1 - 1	
	if2 	= i2 + 1
	while y[if2] >= 0.5*(y[imx]+y[i2]) and if2 < x.shape[0]-2:
		if2 	= if2 + 1
	fwhm1 	= 2*(x[i1]-x[if1])
	fwhm2 	= 2*(x[if2]-x[i2])
	
	if DEBUG:
		plt.close()
		plt.plot(x[imn:(imx+1)],y[imn:(imx+1)],'k.-')
		plt.plot(x[i1],y[i1],'ro',ms=8)
		plt.plot(x[i2],y[i2],'ro',ms=8)		
		plt.plot(x[imn:(imx+1)],np.polyval(pbg,(x[imn:(imx+1)])),'r-',lw=1)
		plt.xlabel('Wavelength (nm)')
		plt.ylabel('Counts (arb)')	
		plt.show()
	else:
		plt.close()
	
	#Fit the doublet
	xi 			= x[imn:(imx+1)]
	yi 			= y[imn:(imx+1)]

	yi 			= yi - np.polyval(pbg, xi)
	#Pick the initial values
	x0 			= np.array([a1,l1,fwhm1,a2,l2,fwhm2,0])	
	bnds 		= [(0.1*a1,10*a1),(x[imn],0.5*(l1+l1)),(0.1*fwhm1,10*fwhm1),(0.1*a2,10*a2),(0.5*(l1+l2),x[imx]),(0.1*fwhm2,10*fwhm2),(-100*a1,100*a1)]
	def FitFxn(x):
		A1 			= np.abs(x[0])
		x1 			= x[1]
		gamma1 		= x[2] / 2.0
		A2 			= np.abs(x[3])
		x2 			= x[4]
		gamma2 		= x[5] / 2.0
		C 			= x[6]
		return (yi - A1 / (1.0 + np.power( (xi-x1)/gamma1, 2) ) - A2 / (1.0 + np.power( (xi-x2)/gamma2, 2) ) - C)
	#Do the fit
	xfit		= spop.leastsq(FitFxn, x0)[0]
	pbg[-1] 	= pbg[-1] + xfit[-1]
	bgfxn 		= lambda x: np.polyval(pbg,x)
	return [imn,imx,Lorentzian(np.abs(xfit[0]), xfit[1], np.abs(xfit[2]), bgfxn),Lorentzian(np.abs(xfit[3]), xfit[4], np.abs(xfit[5]), bgfxn)]			
	
def FitFile(f,tag,STARTI=0,NWINDOW=200,ROLL=-250,DEBUG=True):
	f1 		= LabSpec6(f,tag)
	x 		= f1.GetWavelength()
	y 		= f1.GetCounts() - 980

	ybg 	= GetBackground(x,y,ROLL=ROLL)
	
	ii			= FindPeaks(x,y,DEBUG=DEBUG)
	print "%d peaks" % (ii.shape[0])	
	j 				= 0
	fitA 			= []
	fitL 			= []
	fitQ 			= []
	if not os.path.exists("Fits/%s" % (tag)):
		os.makedirs("Fits/%s" % (tag))
	#If user specifies STARTI, don't start from 0
	if STARTI > 0 and os.path.exists('%s.pkl'% (tag)):
		print "Loading from file!"
		ii 				= ii[STARTI:]	
		ddict 			= dill.load(open('%s.pkl' % (tag),'rb'))	
		fitA 			= ddict['a']
		fitL 			= ddict['l']
		fitQ 			= ddict['q']
	for i in ii:
		mode 		= GetMode(x,y,i)
		if mode == 1:
			[imn,imx,a0,lam0,fwhm0,pbg] 	= FindInitialParameters(x,y,np.gradient(ybg),i)
			xi 			= x[imn:(imx+1)]
			yi 			= y[imn:(imx+1)]
			lor 		= FitLorentzian4(xi,yi,a0,lam0,fwhm0,pbg)
			fwhm 		= lor.GetFWHM()
			if fwhm < 0.02 :
				j 		= j + 1
				continue
			try:
				#fwhm 		= np.sqrt(fwhm**2 - 0.05**2)
				plt.plot(xi,yi,'k.')
				plt.plot(xi,np.polyval(pbg,xi),'b-',lw=1)
				plt.plot(xi,lor.GetFitFunction()(xi),'r-',lw=1.5)
				print "i: %d lam: %0.2f Q: %d" % (j+STARTI, lor.GetLambda0(), lor.GetLambda0()/fwhm)
				plt.xlabel('Wavelength (nm)')
				plt.ylabel('Counts ()')
				plt.grid()
				plt.show(block=False)	
				accept 	= int(raw_input("Accept (1 - Yes, 2 - No) -> "))
				if (accept == 1):
					fitL.append(lor.GetLambda0())
					fitQ.append(lor.GetLambda0()/fwhm)	
					fitA.append(lor.GetA())
					if not DEBUG:
						plt.savefig("Fits/%s/%s-%d.png" % (tag,tag, j+STARTI), bbox_inches='tight')						
				plt.close()
			except:
				pass
			print "\n"
		elif mode == 2:
			print "\tFitting doublet"
			[imn,imx,lor1,lor2] 	= FitDoublet(x,y,i,IDX=75,DEBUG=False)
			pbg 					= lor1.GetBackgroundFunction()
			xi 						= x[imn:(imx+1)]
			yi 						= y[imn:(imx+1)]
			l1 						= lor1.GetLambda0()
			fwhm1 					= np.sqrt(lor1.GetFWHM()**2)#-0.05**2)
			q1 						= l1 / fwhm1
			a1 						= lor1.GetA()
			l2 						= lor2.GetLambda0()
			fwhm2 					= np.sqrt(lor2.GetFWHM()**2)#-0.05**2)
			q2 						= l2 / fwhm2	
			a2 						= lor2.GetA()
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
			accept 	= int(raw_input("Accept (1 - Yes, 2 - No) -> "))
			if (accept == 1):
				fitA.append(a1)
				fitL.append(l1)
				fitQ.append(q1)	
				fitA.append(a2)
				fitL.append(l2)
				fitQ.append(q2)					
				if not DEBUG:
					plt.savefig("Fits/%s/%s-%d.png" % (tag, tag, j+STARTI), bbox_inches='tight')						
			plt.close()			
		dill.dump({'a':fitA,'l':fitL,'q':fitQ},open('%s.pkl' % (tag),'wb'))		
		j 		= j + 1
	return [np.array(fitL), np.array(fitQ) ]
	#plt.plot(x,y,'k-')
	#plt.plot(xlin,yf,'r--',lw=2)
	#plt.plot(x[ii],y[ii],'b.',ms=8)
	#plt.xlabel('Wavelength (nm)')
	#plt.ylabel('Counts ()')
	#plt.plot(fitL,fitQ,'k.',ms=8)
	#plt.show()
	#plt.close()

	
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 18,
	'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(math.sqrt(5)+1)]}	

	dirn 			= "U:/Spectroscopy/QOLab/20170907-A3572-2-RT/"#"U:/Spectroscopy/QOLab/20170720-A3572-1-SiO2-5K/5K/"#"U:/Spectroscopy/QOLab/20170512-A3672-1-PostAnneal/uPL-5K/"#"U:/Spectroscopy/QOLab/20170511-A3572-1-5K-Q-Power/"#"U:/Spectroscopy/QOLab/20170609-A3572-1-SiNx-42s-5K/NoBE/"#
	
	if len(sys.argv) == 3:
		STARTI 		= 0
	elif len(sys.argv) == 4:
		STARTI 		= int(sys.argv[3])
	else:
		print "Need at least 2 arguments python AnalyzeMicrodisk.py <filename>.txt <tag> <START INDEX (OPTIONAL)>"
		sys.exit()
	filen 		= "%s%s.txt" % (dirn, sys.argv[1])
	tag 		= sys.argv[2]	
	
	FitFile(filen,tag,STARTI=STARTI)	

