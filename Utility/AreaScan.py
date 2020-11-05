from uPL import *
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import erf
import scipy.optimize as spop
import matplotlib as mpb

class AreaScan(uPL):

	def __init__(self, filen, tag, sw, intT, grat, lam, dx, dy, Nx, Ny,  LAMTOL=1.0, delim="\t"):
		'''
		Class to encapsulate a 1D spot scan to determine spot size.  Extends uPL.
		New parameters:
		lam: target wavelength [nm]
		dx: x-step size [um]
		dy: y-step size [um]
		DelX: x-scan size [um]
		DelY:  y-scan size [um]
		LAMTOL: interval surrounding lam in which to look for maximum [nm]
		'''
		uPL.__init__(self, filen, tag, sw, intT, grat, delim=delim)
		self.lam0 		= lam 
		self.lamtol 	= LAMTOL
		self.dx 		= dx 
		self.dy 		= dy 
		self.Nx 		= Nx 
		self.Ny 		= Ny
		
		
	def GetLambda(self):
		return self.lam0
		
	def GetDx(self):
		return self.dx 
		
	def GetDy(self):
		return self.dy 
		
	def GetNx(self):
		return self.Nx 
		
	def GetNy(self):
		return self.Ny
		
	def GetX(self):
		#Calculate the position vector
		return np.arange(self.Nx)*self.dx 
		
	def GetY(self):
		#Calculate  the position vector
		return np.arange(self.Ny)*self.dy
		
	def GetIntegratedIntensity(self):
		ctrt 	= np.sum(self.GetCountRate()[:,:-1],0)
		
		return ctrt.reshape(self.Nx,self.Ny).transpose()

	def GetSpectrum(self,xi,yi):
		#Return an individual spectrum at position (xi,yi)
		x 		= self.GetX() 
		y 		= self.GetY()
		[xg,yg] = np.meshgrid(x,y)
		xg 		= xg.flatten() 
		yg 		= yg.flatten()
		ii 		= np.argmin(np.power(xg-xi,2) + np.power(yg-yi,2))
		ctrt 	= self.GetCountRate()[:,ii]		
		return [self.lam,ctrt]
		
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
		I 		= np.mean(ctrt,0) #Return the maximum count rate in the interval 
		
		#Return normalized contrast between -1 and +1
		if NORMALIZE:
			Inorm 	= I - np.mean(I)
			I 		= Inorm / np.max(np.abs(Inorm))
		return I.reshape(self.Nx,self.Ny).transpose()
		
	def PlotProfile(self):
		#Calculate the intensity profile along the x-direction
		lamI 	= np.argmin( np.abs(self.lam - self.lam0) )
		dlam 	= np.abs(self.lam[lamI+1] - self.lam[lamI])
		NLAM 	= int(np.round(20*self.lamtol/dlam))
		ctrt2 	= np.mean(self.GetCountRate()[(lamI-NLAM):(lamI+NLAM),:],0)
		ctrt2 	= ctrt2[:(self.Nx*self.Ny+1)]
		ctrt2 	= ctrt2.reshape(self.Nx,self.Ny).transpose()
		
		dat 	= self.GetProfile()/ctrt2
		x 		= self.GetX()
		y 		= self.GetY()
		[xg,yg] 	= np.meshgrid(x,y)
		dat 		= dat-np.min(np.min(dat))
		dat 		= dat / np.max(np.max(dat))
		lev 		= np.linspace(0,1,11)
		CS 			= plt.contourf(x,y,dat,levels=lev) 
		cbar 		= plt.colorbar(CS)	
		cbar.ax.set_ylabel('Counts')	
		plt.xlabel('x [$\mu$m]')
		plt.ylabel('y [$\mu$m]')
		plt.gca().set_aspect('equal')			
		#plt.show()
	
	def SetLambda(self,lam):
		self.lam0 	= lam 
		
	def SetDx(self,dx):
		self.dx 	= dx 
		
	def SetDy(self,dy):
		self.dy 	= dy 
		
	def SetNx(self,Nx):
		self.Nx 	= Nx 
		
	def SetNy(self,Ny):
		self.Ny 	= Ny
		
	def FitGaussian(self,x0,x1,FORWARD,ax,byRow=False,DEBUG=True):
		#Fit a Gaussian profile, summing the data along axis ax
		assert(ax==0 or ax == 1)
		assert(x1 > x0)
		if ax == 0:
			x 		= self.GetX()
		else:
			x 		= self.GetY()
		if byRow:
			pro 	= self.GetProfile(True)[byRow,:]
		else:
			pro 	= np.sum(self.GetProfile(True),axis=ax)
		x00		= x 
		pro0 	= pro
		assert(x1 <= np.max(x))
		assert(x0 >= np.min(x))
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
			m 	= xf[4]
			return pro - a*erf(mul*(x-x0)/(sig/2)) - off - m*x
			
		xf 	= spop.leastsq(FitFxn,[a0,x0,sig,y0,0])
		xf 	= xf[0]
		if DEBUG:
			plt.plot(x00, pro0, 'ko', ms=8)
			plt.plot(x, pro, 'r.', ms=8, label='Data')
			Gauss 	= lambda xi: xf[0]*erf(mul*(xi-xf[1])/(xf[2]/2)) + xf[3] + xf[4]*xi
			xth 	= np.linspace(np.min(x), np.max(x),100)
			plt.plot(xth, Gauss(xth), 'k-', label='Gaussian fit')
			plt.xlabel('x [$\mu$m]')
			plt.ylabel('Contrast [arb]')
			#if FORWARD:
			#	plt.legend(loc='upper left')
			#else:
			#	plt.legend(loc='upper right')
			plt.grid() 
			plt.show() 
		
		return [xf[2],xf[2]*2*np.sqrt(2*np.log(2))]
		
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(math.sqrt(5)+1)]}	
	matplotlib.rcParams.update(params)	

	file1 	= "U:/Spectroscopy/uPL/20160218-A3405C/A3405C-A1-Scan1-351nm.dat"
	sw 		= 0.05
	tint 	= 0.1 
	grat 	= 2400
	dx 		= 0.1
	dy 		= 0.5
	Nx 		= 141
	Ny 		= 21
	lam 	= 351.3
	sp1 	= AreaScan(file1, 'A3405C A1', sw, tint, grat, lam, dx, dy, Nx, Ny, LAMTOL=0.5)
	sp1.PlotProfile()
	plt.savefig('Scan_2-18-16.pdf',bbox_inches='tight')
	plt.clf()
	#sys.exit()
	[s1,fwhm1] 	= sp1.FitGaussian(0.5,12.5,True,0)
	#plt.savefig('Profile_2-18-16.pdf',bbox_inches='tight')
	plt.clf()
	sys.exit()	
	#print "Row: %d Sigma: %0.3f um FWHM: %0.3f um" % (-1, s1, fwhm1)	 	
	fwhm 	= [] 
	for r in range(sp1.GetNy()):
		[s1,fwhm1] 	= sp1.FitGaussian(6.5,12.0,True,0,byRow=r)
		print "Row: %d Sigma: %0.3f um FWHM: %0.3f um" % (r, s1, fwhm1)	
		fwhm.append(fwhm1)
	fwhm 	= np.array(fwhm)
	print "FWHM: %0.2f +/- %0.2f" % (np.mean(fwhm),np.std(fwhm))
	'''
	[s1,fwhm1] 	= sp1.FitGaussian(1.44,3.27,True)
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)	
	[s2,fwhm2] 	= sp1.FitGaussian(3.84,5.3,False)	
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)
	[s2,fwhm2] 	= sp1.FitGaussian(9.0,10.41,True)	
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s1, fwhm1)	
	[s2,fwhm2] 	= sp2.FitGaussian(6.9,9.9,False)
	print "Sigma: %0.3f um FWHM: %0.3f um" % (s2, fwhm2)	
	'''