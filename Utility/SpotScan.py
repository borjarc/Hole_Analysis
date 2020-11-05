from uPL import *
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import erf
import scipy.optimize as spop
import matplotlib as mpb

class SpotScan(uPL):

	def __init__(self, filen, tag, sw, intT, grat, lam, dx, LAMTOL=1.0, delim="\t"):
		'''
		Class to encapsulate a 1D spot scan to determine spot size.  Extends uPL.
		New parameters:
		lam: target wavelength [nm]
		dx: step size [um]
		LAMTOL: interval surrounding lam in which to look for maximum [nm]
		'''
		uPL.__init__(self, filen, tag, sw, intT, grat, delim=delim)
		self.lam0 		= lam 
		self.lamtol 	= LAMTOL
		self.dx 		= dx 
		
	def GetLambda(self):
		return self.lam0
		
	def GetDx(self):
		return self.dx 
		
	def GetNx(self):
		return self.spectra.shape[1]
		
	def GetX(self):
		#Calculate the position vector
		return np.arange(self.spectra.shape[1])*self.dx 
		
	def GetProfile(self,NORMALIZE=False,DEBUG=False):
		#Calculate the intensity profile along the x-direction
		lamI 	= np.argmin( np.abs(self.lam - self.lam0) )
		dlam 	= np.abs(self.lam[lamI+1] - self.lam[lamI])
		NLAM 	= int(np.round(0.5*self.lamtol/dlam))
		ctrt 	= self.GetCountRate()[(lamI-NLAM):(lamI+NLAM),:]
		if DEBUG:
			mylam 	= self.lam[(lamI-NLAM):(lamI+NLAM)]
			plt.plot(mylam, ctrt[:,0],'k.',ms=8)
			plt.xlabel('Wavelength [nm]')
			plt.ylabel('Count rate [1/s]')
			plt.grid()
			plt.show()
		I 		= np.sum(ctrt,0) #Return the maximum count rate in the interval 
		
		#Return normalized contrast between -1 and +1
		if NORMALIZE:
			Inorm 	= I - np.mean(I)
			return Inorm / np.max(np.abs(Inorm))
		return I
		
	def PlotProfile(self,LINETYPE='k-'):
		ctrst0 	= self.GetProfile(True)
		x 		= self.GetX()
		p1 		= np.polyfit(x, ctrst0, 1)
		ctrst 	= ctrst0 - np.polyval(p1, x)
		ctrst 	= ctrst - np.mean(ctrst)
		ctrst 	= ctrst / np.max(np.abs(ctrst))
		plt.plot(x, ctrst0 ,LINETYPE,lw=1,ms=4,label=self.GetTag())
		plt.xlabel('x [$\mu$m]')
		plt.ylabel('Contrast [arb]')
		plt.grid()
		#plt.show()
		
	def SetLambda(self,lam):
		self.lam0 	= lam 
		
	def SetDx(self,dx):
		self.dx 	= dx 
		
	def FitGaussian(self,x0,x1,FORWARD,DEBUG=True):
		#Fit a Gaussian profile 
		assert(x1 > x0)
		x 		= self.GetX()
		pro 	= self.GetProfile(True)
		assert(x1 < np.max(x))
		assert(x0 > np.min(x))
		i0 		= np.argmin(np.abs(x-x0))
		i1 		= np.argmin(np.abs(x-x1))
		assert(i1 > i0)
		if i1 == (pro.shape[0]-1):
			i1 	= i1 - 1
		x 		= x[i0:i1+1]
		pro 	= pro[i0:i1+1]
		y0 		= np.mean(pro)
		x0 		= x[np.argmin(np.abs(pro))]
		a0 		= np.max(np.abs(pro))
		sig 	= np.abs(x[np.argmax(np.abs(pro))] - x0) / 3.
		if FORWARD:
			mul 	= 1 
		else:
			mul 	= -1 
		def FitFxn(xf):
			a 	= xf[0]
			x0 	= xf[1] 
			sig = xf[2] 
			off = xf[3]
			return pro - a*erf(mul*(x-x0)/(sig/2)) - off
			
		xf 	= spop.leastsq(FitFxn,[a0,x0,sig,y0])
		xf 	= xf[0]
		
		if DEBUG:
			plt.plot(x, pro, 'ko', ms=8, label='Data')
			Gauss 	= lambda xi: xf[0]*erf(mul*(xi-xf[1])/(xf[2]/2)) + xf[3]
			xth 	= np.linspace(np.min(x), np.max(x),100)
			plt.plot(xth, Gauss(xth), 'k-', label='Gaussian fit')
			plt.xlabel('x [$\mu$m]')
			plt.ylabel('Contrast [arb]')
			if FORWARD:
				plt.legend(loc='upper left')
			else:
				plt.legend(loc='upper right')
			plt.grid() 
			plt.show() 
		
		return [xf[2],xf[2]*2*np.sqrt(2*np.log(2))]
		
if __name__=="__main__":
	font = {'family' : 'normal',
			'size'   : 16}
	
	matplotlib.rc('font', **font)

	file1 	= "C:/Users/iroussea/Documents/Spectroscopy/uPL/20160202 - A3405B Vacuum 2/G8E2_50um_500ms_150um_380nm.dat"
	file2 	= "C:/Users/iroussea/Documents/Spectroscopy/uPL/20160202 - A3405B Vacuum 2/H3_Blk_Scan_50um_150nm_1s.dat"
	sw 		= 0.05 
	tint 	= 20 
	grat 	= 1200
	dx 		= 0.15
	lam 	= 363.5
	sp1 	= SpotScan(file1, '2.2.16 N$_2$', sw, tint, grat, lam, dx)
	sp2 	= SpotScan(file2, '2.2.16 Vacuum', sw, tint, grat, lam, dx)	
	sp1.PlotProfile('k-o')
	sp2.PlotProfile('b-o')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()
	[s1,fwhm1] 	= sp1.FitGaussian(5.0,6.9,True)
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)	
	[s1,fwhm1] 	= sp1.FitGaussian(1.44,3.27,True)
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)	
	[s2,fwhm2] 	= sp1.FitGaussian(3.84,5.3,False)	
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)
	[s2,fwhm2] 	= sp1.FitGaussian(9.0,10.41,True)	
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)	
	[s2,fwhm2] 	= sp2.FitGaussian(6.9,9.9,False)
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s2, fwhm2)	