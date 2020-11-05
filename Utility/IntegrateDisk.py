import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spsp
import scipy.integrate as spi 
import scipy.optimize as spop

def AiryDisk(x):
	return 4*np.power(spsp.jn(1,x) / x,2)
	
def AiryDiskXY(x,y):
	r 		= np.sqrt(np.power(x,2)+np.power(y,2))
	return AiryDisk(r)
	
def IntegrateAiryDisk(x0,Rinf=20):
	z 		= np.zeros(x0.shape[0])
	i 		= 0 
	for xi in x0:
		print i
		z[i] 	= spi.quad(AiryDisk,x0[i],Rinf)[0]
		i 	= i + 1
	return z	
	
def IntegrateAiryDisk2D(x0,Rinf=20):
	z 		= np.zeros(x0.shape[0])
	glo 	= lambda x: -np.sqrt(Rinf**2 - np.power(x,2))
	ghi 	= lambda x: np.sqrt(Rinf**2 - np.power(x,2))
	i 		= 0 
	for xi in x0:
		print i
		z[i] 	= spi.dblquad(AiryDiskXY,x0[i],Rinf,glo,ghi)[0]
		i 	= i + 1
	return z

def FitGaussian(x,pro,x0,x1,FORWARD,DEBUG=True):
	#Fit a Gaussian profile, summing the data along axis ax
	x00		= x 
	pro0 	= pro
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
		return pro - a*spsp.erf(mul*(x-x0)/(sig/2)) - off
		
	xf 	= spop.leastsq(FitFxn,[a0,x0,sig,y0])
	xf 	= xf[0]
	if DEBUG:
		plt.plot(x00, pro0, 'ko', ms=8)
		plt.plot(x, pro, 'r.', ms=8, label='Data')
		Gauss 	= lambda xi: xf[0]*spsp.erf(mul*(xi-xf[1])/(xf[2]/2)) + xf[3]
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
	x 	= np.linspace(-10,10,100)
	y 	= IntegrateAiryDisk(x)
	[s1,fwhm1] 	= FitGaussian(x,y,-10,10,False)
	print "Sigma: %0.3f FWHM: %0.3f" % (s1, fwhm1)		