import numpy as np
import scipy.interpolate as spip
from numpy.random import normal
from uPL import *
import cPickle as pickle
import datetime
from datetime import date

class UncertaintyLUT:
	'''
	Create lookup tables for the uncertainty in line position, linewidth, and amplitude
	as a function of the signal-to-noise ratio, defined as the ratio of the peak signal 
	amplitude to the standard deviation of the Gaussian noise distribution.  Use the Monte
	Carlo method.
	
	Use logarithmic spacing for the signal-to-noise ratio and linear spacing for the 
	linewidth
	
	Input parameters:
		SNmin : minimum signal-to-noise ratio
		SNmax : maximum signal-to-noise ratio
		lwMin : minimum linewidth [nm]
		lwMax : maximum linewidth [nm]
		Nsn : Number of signal-to-noise table points
		Nlw : Number of linewidth table points
	'''
	def __init__(self,SNmin,SNmax,lwMin,lwMax,NTEST,Nsn=80,Nlw=40):
		self.SNmin 		= SNmin 
		self.SNmax 		= SNmax 
		self.lwMin 		= lwMin
		self.lwMax 		= lwMax
		self.Ntest	 	= int(NTEST)
		self.Nsn 		= Nsn 
		self.Nlw 		= Nlw
		self.__RecalculateLUT__()
		
	def __RecalculateLUT__(self):
		[lwG,snG] 	= np.meshgrid(np.linspace(self.lwMin,self.lwMax,self.Nlw),\
		#np.linspace(self.SNmin,self.SNmax))
		np.logspace(np.log10(self.SNmin), np.log10(self.SNmax),self.Nsn))
		A 			= np.zeros(lwG.shape,dtype='float64')
		FWHM 		= np.zeros(lwG.shape,dtype='float64')
		wl			= np.zeros(lwG.shape,dtype='float64')
		stdA 		= np.zeros(lwG.shape,dtype='float64')
		stdFWHM 	= np.zeros(lwG.shape,dtype='float64')
		stdWL 		= np.zeros(lwG.shape,dtype='float64')		
		for i in np.arange(lwG.shape[0]):
			for j in np.arange(lwG.shape[1]):
				#Calculate the standard deviation of the current matrix element over NTEST samples
				lw 	= lwG[i,j]
				sn 	= snG[i,j]
				print "LW: %0.3f S/N: %0.1f" % (lw, sn)
				[Ab,FWb,lamB,As,FWs,lamS] 	= self.__SimulateFits__(lw,sn,self.Ntest)
				A[i,j] 		= Ab - sn
				FWHM[i,j] 	= FWb - lw
				wl[i,j] 	= lamB 
				stdA[i,j]	= As 
				stdFWHM[i,j] 	= FWs
				stdWL[i,j] 		= lamS
		self.lw 		= lwG
		self.sn 		= snG
		self.Adiff 		= A 
		self.FWHMdiff 	= FWHM
		self.WLdiff 	= wl 
		self.Astd 		= stdA 
		self.FWHMstd 	= stdFWHM
		self.WLstd 		= stdWL
		#Write the interpolation objects
		self.FWHMi 		= spip.interp2d(lwG,snG,stdFWHM)
		self.Ai 		= spip.interp2d(lwG,snG,stdA)
		self.WLi 		= spip.interp2d(lwG,snG,stdWL)
		
	def __SimulateFits__(self,lw,sn,NT,dlam=0.00793404982902,NFWHM=20,DEBUG=True):
		a 			= np.zeros(NT)
		fwhm 		= np.zeros(NT)
		lam 		= np.zeros(NT)
		for i in np.arange(NT):
			uplobj 	= self.__MakeNoisyUPL__(lw,sn,dlam,NFWHM)
			lor	 	= FitLorentzian2(uplobj, 0, lw, np.zeros(2))
			a[i] 	= lor.GetA()
			fwhm[i]	= lor.GetFWHM()
			lam[i]	= lor.GetLambda0()
		if DEBUG:
			pass
		return [np.mean(a),np.mean(fwhm),np.mean(lam), \
		np.std(a),np.std(fwhm),np.std(lam)]
				
	def __MakeNoisyUPL__(self,lw,sn,dlam,NFWHM,DEBUG=False):
		lam 		= dlam*np.arange(-np.floor(0.5*NFWHM*lw/dlam),np.ceil(0.5*NFWHM*lw/dlam)+1)
		s 			= normal(0,1.0,lam.shape[0]) #Use a standard normal distribution
		lth 		= Lorentzian(sn, 0., lw, np.zeros([0,0]))
		lfit 		= lth.GetFitFunction()
		if DEBUG:
			plt.plot(lam,s,'r.')
			plt.plot(lam,lfit(lam),'b.')
			plt.plot(lam,s+lfit(lam),'r.')
			plt.xlabel('Wavlelength [nm]')
			plt.ylabel('Count rate [1/s]')
			plt.show()
			sys.exit()
		return uPLLUT( lam, s+lfit(lam), 'test', 50, 1, 2400 )

	#Evaluate the 2D interpolation objects
	def EstimateFWHM(self, lw, sn, sig):
		return np.array( map( lambda lwi, sni: self.FWHMi(lwi, sni/sig), lw, sn)).flatten()
		
	def EstimateA(self, lw, sn, sig):
		return np.array( map( lambda lwi, sni: self.Ai(lwi, sni/sig), lw, sn)).flatten()
		
	def EstimateWL(self, lw, sn, sig):
		return np.array( map( lambda lwi, sni: self.WLi(lwi, sni/sig), lw, sn)).flatten()
		
	def PlotFWHMdiff(self):
		self.__MakeContourf__(self.FWHMdiff,'$\hat{\Delta\lambda}-\Delta\lambda$')
	
	def PlotFWHMstd(self):
		self.__MakeContourf__(self.FWHMstd,'$\sigma_{\Delta\lambda}$')
	
	def PlotWLdiff(self):
		self.__MakeContourf__(self.WLdiff,'$\hat{\lambda}-\lambda$')
	
	def PlotWLstd(self):
		self.__MakeContourf__(self.WLstd,'$\sigma_{\lambda}$')
	
	def PlotAdiff(self):
		self.__MakeContourf__(self.Adiff,'$\hat{A}-A$')
		
	def PlotAstd(self):
		self.__MakeContourf__(self.Astd,'$\sigma_A$')
		
	def __MakeContourf__(self,z,title):
		CS = plt.contourf(self.lw, np.log10(self.sn), z, cmap=plt.cm.afmhot)
		plt.xlabel('FWHM [nm]')
		plt.ylabel('log$_{10}$(S/N)')
		plt.title(title)
		plt.colorbar(CS)
		stmp				= datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S")
		plt.savefig("FitStatistics/%s.png" % (stmp))
		plt.savefig("FitStatistics/%s.eps" % (stmp))
		plt.clf()
										
	def GetMinLW(self):
		return self.LWmin
		
	def GetMaxLW(self):
		return self.LWmax
		
	def GetMinSN(self):
		return self.SNmin
		
	def GetMaxSN(self):
		return self.SNmax
		
	def GetNumSN(self):
		return self.Nsn 
		
	def GetNumLW(self):
		return self.Nlw
		
	def GetNtest(self):
		return self.Ntest

	def SetMinLW(self, LWmin):
		self.LWmin 		= LWmin
		self.__RecalculateLUT__()
		
	def SetMaxLW(self, LWmax):
		self.LWmax 		= LWmax
		self.__RecalculateLUT__()		
		
	def SetMinSN(self, SNmin):
		self.SNmin 		= SNmin
		self.__RecalculateLUT__()		
		
	def SetMaxSN(self, SNmax):
		self.SNmax 		= SNmax
		self.__RecalculateLUT__()
		
	def SetNumSN(self,NumSN):
		self.Nsn 	= NumSN	
		self.__RecalculateLUT__()
		
	def SetNumLW(self,NumLW):
		self.Nlw 	= NumLW	
		self.__RecalculateLUT__()	
		
	def SetNtest(self,Ntest):
		self.Ntest 	= Ntest
		self.__RecalculateLUT__()	
	
if __name__=="__main__":
	SNmin 	 	= 5
	SNmax 		= 3000
	lwMin 		= 0.05
	lwMax 		= 0.7
	Ntest 		= 200
	ulut		= UncertaintyLUT(SNmin,SNmax,lwMin,lwMax,Ntest)
	ulut.PlotFWHMdiff()
	ulut.PlotFWHMstd()
	ulut.PlotWLdiff()
	ulut.PlotWLstd()
	ulut.PlotAdiff()
	ulut.PlotAstd()
	pickle.dump(ulut,open('uLUT_sig1_2.pkl','wb'))