import numpy as np
import scipy.interpolate as spip
import scipy.optimize as spop
import matplotlib
import matplotlib.pyplot as plt
import math
import os.path 
import sys
import csv
from numpy.fft import *
from scipy import stats
from scipy import signal
from datetime import datetime
import copy

class uPL:

	def __init__(self, filen, tag, sw, intT, grat, delim="\t"):
		'''
		Create an object representing spectra recorded using the LabView program on
		the uPL setup.
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		sw: 		slit width [mm]
		intT:		integration time [s]
		grat: 		grating ruling [lp/mm]
		dx: 		x-step [um]
		dy: 		y-step [um]
		'''
		assert(os.path.exists(filen))
		self.tag 	= tag
		self.sw 	= sw 
		self.intT 	= intT 
		self.grat 	= grat 
		[self.lam, self.spectra] 	= self.__LoadFile__(filen,delim)
			
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 		= 0
			wl 		= []
			cts 	= []		
			for row in spamreader:
				wl.append(float(row[0]))
				subCts 	= []
				for i in range(1,len(row)):
					try:
					 	ele 	= float(row[i])
						subCts.append(ele)
					except:
						pass
				if cts == []:
					cts 	= np.array(subCts)
				else:
					cts 	= np.vstack([cts, np.array(subCts)])
				j 			= j + 1
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			if cts.shape[1] == 1:
				cts 	= cts[:,0]
			return [wl, cts]
			
	def GetSlitWidth(self):
		return self.sw 
		
	def GetIntegrationTime(self):
		return self.intT
		
	def GetGrating(self):
		return self.grat
		
	def GetTag(self):
		return self.tag 
		
	def SetSlitWidth(self,sw):
		assert(sw > 0)
		self.sw 		= sw 
		
	def SetIntegrationTime(self, intT):
		assert(intT > 0)
		self.intT 		= intT 
		
	def SetGrating(self, grat):
		assert(grat > 0)
		self.grat 	= grat
		
	def SetTag(self, tag):
		self.tag 	= tag

	def GetWavelength(self): 
		return self.lam 
		
	def GetCountRate(self):
		return self.spectra/self.intT 
		
	def GetCounts(self):
		return self.spectra 
		
	def SetCounts(self,ctrt):
		self.spectra 	= ctrt 
		
	def SetWavelength(self,wl):
		self.lam 		= wl
		
class LabSpec6(uPL):
	def __init__(self, filen, tag, delim="\t", SPECIAL=False):
		'''
		Create an object representing a spectrum recorded using the LabSpec6
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))
		[self.lam, self.spectra, self.intT, self.sw, self.grat, self.lam0, self.N, self.timestamp] 	= self.__LoadFile__(filen,delim,SPECIAL)
		self.tag 	= tag
		
	def GetFilteredCounts(self,f0=2507.,Q=2.5):
		y 		= self.spectra 
		x 		= 1240./self.lam #(eV)
		fs 		= 1. / np.abs(np.mean(x[1:]-x[:-1])) #(1/eV)
		w0 		= f0 / (fs / 2.)
		b,a 	= signal.iirnotch(w0,Q)
		return signal.lfilter(b,a,y)
		
	def GetFilteredCountRate(self):
		return self.GetFilteredCounts() / self.intT
		
	def GetCenterWavelength(self):
		return self.lam0 
		
	def GetAccumulations(self):
		return self.N
		
	def GetTimestamp(self):
		return self.timestamp
			
	def __LoadFile__(self, filen, delim, SPECIAL):
		grat 	 = 0 
		lam0 	 = 0 
		sw 		 = 0 
		intT 	 = 0
		N 		 = 0 
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []		
			for row in spamreader:
				if "#" in row[0]:
					key 	= row[0].split("#")[-1].split("=")[0]
					if key == "Acq. time (s)":
						intT 	= float(row[-1].replace(",","."))
					elif key == "Accumulations":
						N 		= int(row[-1])
					elif key == "Dark correction":
						pass
					elif key == "Grating":
						grat 	= int(row[-1].split("gr/mm")[0])
					elif key == "Spectrometer (nm)":
						lam0 	= float(row[-1].replace(",","."))
					elif "Front entrance slit" in key:
						sw 		= float(row[-1].replace(",",".")) / 1000.
					elif key == "Acquired":
						timestamp 	= datetime.strptime(row[-1], "%d.%m.%Y %H:%M:%S")
				else:
					if SPECIAL:
						wl.append(float(row[0])+0.001*float(row[1]))
					else:
						wl.append(float(row[0].replace(",",".")))
					subCts 	= []
					for i in range(1,len(row)):
						try:
							if SPECIAL:
								ele 	= float(row[2])
							else:
								ele 	= float(row[i].replace(",","."))
							subCts.append(ele)
						except:
							pass
					if cts == []:
						cts 	= np.array(subCts)
					else:
						cts 	= np.vstack([cts, np.array(subCts)])
					j 			= j + 1
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			if cts.shape[1] == 1:
				cts 	= cts[:,0]
			return [wl, cts, intT, sw, grat, lam0, N, timestamp]
			
class LS_Calibration(LabSpec6):
	def __init__(self, filen, tag, delim="\t"):
		'''
		Create an object representing a calibration spectrum recorded using the LabSpec6.
		User then adds lines using AddLine and then sends wavelength vectors to be corrected into CorrectWavelength
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))
		LabSpec6.__init__(self, filen, tag, delim)
		pks 			= IdentifyPeaks(self,exclWL=[],qmin=1000,qmax=50000,DEBUG=True,SIGCUT=0.5)
		self.FitPeaks 	= np.sort(np.array(map(lambda x: x.GetLambda0(), pks)))
		self.lines 		= []
		self.p 			= None 
		self.Unchanged 	= True
		
	def AddLine(self,line):
		self.lines.append(line)
		self.Unchanged 	= False
		
	def GetFitPeaks(self):
		return self.FitPeaks
		
	def __CalculateCorrectionPolynomial__(self, pcorr=2, DEBUG=True, WLTOL=2. ):
		#Find the closest fit lines for each given line
		fits 		= self.FitPeaks
		lines 		= np.sort(np.array(self.lines))
		#Need lines 
		assert(lines.shape[0] > 1)
		lineMat 	= np.outer(np.ones(fits.shape[0]),lines)
		fitMat		= np.outer(fits,np.ones(lines.shape[0]))
		dMat 		= np.abs(lineMat-fitMat)
		ii 			= np.argmin(dMat,0)
		selFits 	= fits[ii]
		p 			= np.polyfit(lines,selFits,pcorr)
		np.min(dMat,0)
		assert(np.all(np.min(dMat,0) <= WLTOL))
		if DEBUG:
			plt.plot(lines,selFits,'ko',ms=8)
			plt.xlabel('Wavelength recorded (nm)')
			plt.ylabel('Actual wavelength (nm)')
			plt.grid()
			plt.show()			
			plt.plot(self.GetWavelength(),self.GetWavelength()-np.polyval(p,self.GetWavelength()))
			plt.xlabel('Wavelength recorded (nm)')
			plt.ylabel('$\Delta$ wavelength (nm)')
			plt.grid()
			plt.show()
		return p
	
	def CorrectWavelength(self,wlI):
		if self.p is None or not self.Unchanged():
			self.p 	= self.__CalculateCorrectionPolynomial__()
			self.Unchanged = True 
		return np.polyval(self.p, wlI)

class LS_CalibrationTwoPoint(LabSpec6):
	def __init__(self, filen, tag, delim="\t"):
		'''
		Create an object representing a calibration spectrum recorded using the LabSpec6.
		User then adds lines using AddLine and then sends wavelength vectors to be corrected into CorrectWavelength
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))
		LabSpec6.__init__(self, filen, tag, delim)
		pks 			= IdentifyPeaks(self,exclWL=[],qmin=1000,qmax=50000,DEBUG=False,SIGCUT=0.5,NFWHM=0.5)
		self.FitPeaks 	= np.sort(np.array(map(lambda x: x.GetLambda0(), pks)))
		self.lines 		= []
		self.sLines 	= [] #Lines in the actual spectrum 
		self.p 			= None 
		self.Unchanged 	= True
		
	def AddLine(self,line,line2):
		self.lines.append(line)
		self.sLines.append(line2)
		self.Unchanged 	= False
		
	def GetFitPeaks(self):
		return self.FitPeaks
		
	def __CalculateCorrectionPolynomial__(self, pcorr=2, DEBUG=False, WLTOL=0.1 ):
		#Find the closest fit lines for each given line
		fits 		= self.FitPeaks
		calLines 	= np.sort(np.array(self.lines))
		lines 		= np.sort(np.array(self.sLines))
		#Need lines 
		assert(lines.shape[0] > 1)
		lineMat 	= np.outer(np.ones(fits.shape[0]),lines)
		fitMat		= np.outer(fits,np.ones(lines.shape[0]))
		dMat 		= np.abs(lineMat-fitMat)
		ii 			= np.argmin(dMat,0)
		jj 			= np.nonzero(np.min(dMat,0)<=WLTOL)[0]
		ii 			= ii[jj] #Restrict to smallest shifts
		assert(ii.shape[0] >= pcorr) #Make sure we can find the fits
		selFits 	= fits[ii]
		deltaWL 	= calLines[jj] - selFits
		p 			= np.polyfit(selFits,deltaWL,pcorr)
		#assert(np.all(np.min(dMat,0) <= WLTOL))
		if DEBUG:
			plt.plot(lines[jj],selFits,'ko',ms=8)
			plt.xlabel('Wavelength recorded (nm)')
			plt.ylabel('Actual wavelength (nm)')
			plt.grid()
			plt.show()			
			plt.plot(self.GetWavelength(),np.polyval(p,self.GetWavelength()))
			plt.xlabel('Wavelength recorded (nm)')
			plt.ylabel('$\Delta$ wavelength (nm)')
			plt.grid()
			plt.show()
		return p
	
	def CorrectWavelength(self,wlI):
		if self.p is None or not self.Unchanged():
			self.p 	= self.__CalculateCorrectionPolynomial__()
			self.Unchanged = True 
		return np.polyval(self.p, wlI) + wlI	
		
class HS_Mapscan(uPL):
	'''
	For Hyperspectral mapscan
	'''
	def __init__(self,filen,sw,tag,delim=','):
		assert(os.path.exists(filen))
		self.sw 	= sw 
		self.tag 	= tag	
		[self.lam, self.spectra, self.x, self.y, self.grat, self.intT] 	= self.__LoadFile__(filen,delim)

	def __LoadFile__(self, filen, delim):
		grat 	 = 0 
		lam0 	 = 0 
		intT 	 = 1.0
		N 		 = 0 
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []
			x 			= []
			y 			= []
			row 	= spamreader.next()
			assert("MapscanFHR" in row[0])
			while not ( "Grating" in row[0]):
				row 	= spamreader.next()
			grat 		= int(row[0].split(":")[1])
			intT 		= int(row[3].split("(ms):")[1])/1000.
			while not ("x(um)" in row[0]):
				row 	= spamreader.next()	
			row 		= spamreader.next()
			for j in range(2,len(row)):
				wl.append(float(row[j]))
			for row in spamreader:
				if len(row) == (len(wl) + 2):
					x.append(float(row[0]))	
					y.append(float(row[1]))
					ctstmp 	= []
					for j in range(2,len(row)):
						ctstmp.append(float(row[j]))
					cts.append(ctstmp)
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			x 	 		= np.array(x)
			y 			= np.array(y)
			ii 			= np.argsort(wl)
			return [wl,cts,x,y,grat,intT]		

	def GetX(self):
		return self.x 
		
	def GetY(self):
		return self.y
		
	def GetSingleSpectrum(self,x,y):
		if x in self.x and y in self.y:
			i 		= np.nonzero(np.logical_and(self.x == x, self.y == y))[0][0]
			return self.spectra[i,:]
		else:
			return None
			
	def GetSingleUPL(self,x,y):
		'''
		A bit of a hack, but returns single uPL object with only the spectrum at a specific point.  None if error.
		'''
		upltmp 	= copy.deepcopy(self)
		spec 	= self.GetSingleSpectrum(x,y)
		if spec is not None:
			upltmp.SetCounts(spec)
			return upltmp
		else:
			return None
			
	def GetMonochromatic(self, wl):
		if wl in self.lam:
			j 	= np.argmin(np.abs(self.lam - wl))
			return self.spectra[:,j]
		else:
			return None
			
	def GetIntensity(self):
		return np.sum(self.spectra,1)
			
class PowerSweep(uPL):
	'''
	For JYWaterfall
	'''
	def __init__(self,filen,sw,grat,tag,delim=','):
		assert(os.path.exists(filen))
		self.intT 	= 1.0
		self.sw 	= sw 
		self.grat 	= grat
		self.tag 	= tag		
		[self.lam, self.spectra, self.power, self.units, self.intT, self.scaling] 	= self.__LoadFile__(filen,delim)

	def __LoadFile__(self, filen, delim):
		grat 	 = 0 
		lam0 	 = 0 
		intT 	 = 1.0
		N 		 = 0 
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []
			pwrs 		= []
			row 	= spamreader.next()
			while not ( "Units" in row[0]):
				row 	= spamreader.next()
			units 		= row[0].split(": ")[1]
			intT 		= int(row[1].split("(ms):")[1])/1000.
			scaling 	= not("OFF" in row[2].split("scaling:")[1])
			row 		= spamreader.next()
			#Get the wavelengths
			row 		= spamreader.next()
			for j in range(1,len(row)):
				wl.append(float(row[j]))
			for row in spamreader:
				if len(row) == (len(wl) + 1):
					pwrs.append(float(row[0]))	
					ctstmp 	= []
					for j in range(1,len(row)):
						ctstmp.append(float(row[j]))
					cts.append(ctstmp)
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			pwrs 	 	= np.array(pwrs)
			ii 			= np.argsort(wl)
			return [wl,cts,pwrs,units,intT,scaling]		

	def GetPowers(self):
		return self.power 
		
	def GetData(self):
		return self.spectra
	
	def SetData(self,dat):
		self.spectra =	 dat
		
	def SetPowers(self,power):
		self.power = power
		
	def GetSingleSpectrum(self,pwr):
		if pwr in self.power:
			i 		= np.nonzero(self.power == pwr)[0][0]
			return self.spectra[i]
		else:
			return None
			
	def GetScaling(self):
		return self.scaling 
		
	def GetUnits(self):
		return self.units
			
	def GetSingleUPL(self,pwr):
		'''
		A bit of a hack, but returns single uPL object with only the power at a specific point.  None if error.
		'''
		upltmp 	= copy.deepcopy(self)
		spec 	= self.GetSingleSpectrum(pwr)
		if spec is not None:
			upltmp.SetCounts(spec)
			return upltmp
		else:
			return None
			
class TimeSeries(PowerSweep):
	'''
	For JYWaterfall
	'''
	def __init__(self,filen,sw,grat,tag,delim=','):
		assert(os.path.exists(filen))
		PowerSweep.__init__(self, filen,sw,grat,tag,delim)

	def GetTimes(self):
		return self.power 
		
class SPEX(uPL):
	def __init__(self, filen, sw, grat, tag, Nav, delim=","):
		'''
		Create an object representing a spectrum recorded using LabView for the HBT setup
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))
		self.intT 	= 1 
		self.sw 	= sw 
		self.grat 	= grat
		self.N	= Nav
		[self.lam, self.spectra] 	= self.__LoadFile__(filen,delim)
		self.tag 	= tag
		
	def GetAccumulations(self):
		return self.N
			
	def __LoadFile__(self, filen, delim):
		grat 	 = 0 
		lam0 	 = 0 
		sw 		 = 0 
		intT 	 = 0
		N 		 = 0 
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []
			row 		= spamreader.next()
			while not (row[0] == "X_Value"):
				row 	= spamreader.next()
			for row in spamreader:
				if len(row) == 3:
					wl.append(float(row[1]))
					cts.append(float(row[2]))
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			ii 			= np.argsort(wl)
			return [wl[ii], cts[ii]]
		
class uPLLUT(uPL):
	'''
	Make a uPL class derivative where you give a count # and wavelength as input parameters.
	'''
	def __init__(self, wl, cts, tag, sw, intT, grat):
		self.lam 		= wl 
		self.spectra 	= cts
		self.tag 		= tag
		self.sw 		= sw 
		self.intT 		= intT 
		self.grat 		= grat 		
		
class Gaussian:
	'''
	Define a class to hold Lorentzian function & fit parameters
	'''
	def __init__(self, a, lam0, fwhm):
		self.A 			= a 
		self.lam0 		= lam0 
		self.fwhm 		= fwhm 
		
	def GetFitFunction(self):
		def BestFitGaussian(lam):		
			sig 		= self.fwhm / (2*np.sqrt(2*np.log(2)))
			return np.abs(self.A) * np.exp( -np.power( (lam-self.lam0) / (np.sqrt(2)*sig), 2) )
		return BestFitGaussian
		
	def GetA(self):
		return self.A 
		
	def GetLambda0(self):
		return self.lam0 
		
	def GetFWHM(self):
		return self.fwhm 
	
	def SetA(self,A):
		self.A  = A
		
	def SetLambda0(self,lam0):
		self.lam0  = lam0 
		
	def SetFWHM(self,fwhm):
		self.fwhm  = fwhm 
		
class Fano:
	'''
	Define a class to hold Lorentzian function & fit parameters
	'''
	def __init__(self, a, lam0, fwhm, q):
		self.A 			= a 
		self.lam0 		= lam0 
		self.fwhm 		= fwhm
		self.q 			= q
		
	def GetFitFunction(self):
		def BestFitFano(lam):		
			sig 		= self.fwhm / 2
			return np.abs(self.A) * np.power(self.q*sig+lam-self.lam0,2) / \
			( np.power(sig, 2) + np.power(lam-self.lam0, 2) )
		return BestFitFano
		
	def GetA(self):
		return self.A 
		
	def GetLambda0(self):
		return self.lam0 
		
	def GetFWHM(self):
		return self.fwhm 
		
	def GetQ(self):
		return self.q 
	
	def SetA(self,A):
		self.A  = A
		
	def SetLambda0(self,lam0):
		self.lam0  = lam0 
		
	def SetFWHM(self,fwhm):
		self.fwhm  = fwhm 		

	def SetQ(self,q):
		self.q 		= q		
		
class Lorentzian:
	'''
	Define a class to hold Lorentzian function & fit parameters
	'''
	def __init__(self, a, lam0, fwhm, p_bgd, noise=0):
		self.A 			= a 
		self.lam0 		= lam0 
		self.fwhm 		= fwhm 
		self.p_bgd 		= p_bgd
		self.noise 		= noise 
		self.fwhm0 		= 1.
		
	def GetFitFunction(self):
		def BestFitLorentzian(lam):		
			bgfn 	= self.GetBackgroundFunction()
			return np.abs(self.A) / (1.0 + np.power( (lam-self.lam0)/(self.fwhm/2.0), 2)) + bgfn(lam)
		return BestFitLorentzian
		
	def GetA(self):
		return self.A 
		
	def GetLambda0(self):
		return self.lam0 
		
	def GetFWHM(self):
		return self.fwhm 
		
	def GetBackgroundFunction(self):
		if isinstance( self.p_bgd	, (np.ndarray, np.generic) ):
			return lambda x: np.polyval( self.p_bgd	,x)
		else:
			return self.p_bgd	
	
	def GetNoise(self):
		return self.noise 
		
	def SetA(self,A):
		self.A  = A
		
	def SetLambda0(self,lam0):
		self.lam0  = lam0 
		
	def SetFWHM(self,fwhm):
		self.fwhm  = fwhm 
		
	def SetBackground(self,p_bgd):
		self.p_bgd 	= p_bgd
		
	def SetNoise(self,noise):
		self.noise 	= noise 
		
	def SetFWHM0(self,fwhm0):
		self.fwhm0 	= fwhm0 
		
	def GetFWHM0(self):
		return self.fwhm0
		
def FitFano(uplobj, lam0, fwhm0, q0=0, DEGBKGD=0, NFWHM=30, IDX=None):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	p_bgd:  Background level polynomial fit
	q0: 	Asymmetry parameter for Fano fit 
	NFWHM: 	Number of FWHM on either side of lam0 to keep data
	Returns:
	lfxn: 	Best-fit Lorentzian function that takes an array of wavelengths [nm] as input
	lamBF: 	Best-fit center wavelength [nm]
	fwhmBF: Best-fit FWHM [nm]
	ABF: Best-fit max count rate [1/s]
	'''
	
	lam 	= uplobj.GetWavelength()
	ctrt 	= uplobj.GetCountRate()	

	#Pick the initial values
	ii 			= np.nonzero(np.logical_and( lam >= (lam0-0.5*NFWHM*fwhm0), lam <= (lam0+0.5*NFWHM*fwhm0)))[0]
	bkgd 		= stats.mode(np.round(ctrt),axis=None)[0][0]
	pbkd 		= np.array([bkgd])
	bgfxn 		= lambda x: np.polyval(pbkd, x)
	iinterp 	= spip.interp1d(lam,ctrt-bgfxn(lam))
	x0 			= [np.max(ctrt)-bkgd,lam0,fwhm0,q0]
	x0 			= np.array(x0)
	ctrt 		= ctrt - np.min(ctrt)
	def FitFxn(x):
		A 		= x[0]
		x0 		= x[1]
		fwhm 	= x[2]
		q 		= x[3]
		fano 	= Fano(A, x0, fwhm, q).GetFitFunction()
		return ctrt[ii] - fano(lam[ii])

	#Do the fit
	xfit 	= spop.leastsq(FitFxn, x0)[0]
	return [Fano(xfit[0], xfit[1], xfit[2], xfit[3]), np.array([bkgd])]

def GetBackground1(x,y,norm_pass=0.005,norm_stop=0.05,ROLL=0):
	#Fit the background
	fitp			= spip.interp1d(x,y,kind='linear')
	xlin 			= np.linspace(np.min(x),np.max(x),x.shape[0])
	ylin 			= fitp(xlin)	
	#Apply a low-pass filter to get rid of peaks
	(b, a) 		= signal.butter(4,0.005,btype='low')
	return np.roll(signal.lfilter(b, a, ylin),ROLL)
	
def FitLorentzian(uplobj, lam0, fwhm0, NFWHM=15):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	NFWHM: 	Number of FWHM on either side of lam0 to keep data
	Returns:
	lfxn: 	Best-fit Lorentzian function that takes an array of wavelengths [nm] as input
	lamBF: 	Best-fit center wavelength [nm]
	fwhmBF: Best-fit FWHM [nm]
	ABF: Best-fit max count rate [1/s]
	'''
	lam 	= uplobj.GetWavelength()
	ctrt 	= uplobj.GetCountRate()
	ctrt0 	= ctrt.copy()	

	#for i in range(ctrt0.shape[1]):
	ctrt 	= ctrt0#[:,i]
	bg0 	= stats.mode(np.round(ctrt),axis=None)[0][0]
	ii 		= np.nonzero(np.logical_and( lam >= lam0-NFWHM*fwhm0, lam <= lam0+NFWHM*fwhm0 ))[0]	
	def FitFxn(x):
		A 		= np.abs(x[0])
		x0 		= x[1]
		gamma 	= x[2] / 2.0
		bg 		= x[3] 
		m 		= x[4]
		return (ctrt[ii] - A / (1.0 + np.power( (lam[ii]-x0)/gamma, 2) ) - bg - m*lam[ii]) / ii.shape[0]
		#return np.sum(np.power(ctrt[ii] - A / (1.0 + np.power( (lam[ii]-x0)/gamma, 2) ) - bg - m*lam[ii],2))
	#Pick the initial values
	ii 		= np.nonzero(np.logical_and( lam >= (lam0-NFWHM*fwhm0), lam <= (lam0+NFWHM*fwhm0)))[0]
	lamii 	= lam[ii]
	#p1 		= np.polyfit(lamii,ctrt[ii],1)
	x0 		= np.array([np.max(ctrt[ii])-bg0,lamii[np.argmax(ctrt[ii])],fwhm0,bg0,0])
	#Do the fit
	xfit 	= spop.leastsq(FitFxn, x0)[0]
	#xfit,_,_ 	= spop.fmin_l_bfgs_b(FitFxn, x0, approx_grad=True, bounds=[(0,10*x0[0]),(x0[1]-1,x0[1]+1),(0.001,10*fwhm0),(0,bg0),(-1e4,1e4)])
	return Lorentzian(xfit[0], xfit[1], xfit[2], np.array([xfit[4], xfit[3]]))
	
def FitLorentzian2(uplobj, lam0, fwhm0, p_bgd, NFWHM=15, IDX=None):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	p_bgd:  Background level polynomial fit
	NFWHM: 	Number of FWHM on either side of lam0 to keep data
	Returns:
	lfxn: 	Best-fit Lorentzian function that takes an array of wavelengths [nm] as input
	lamBF: 	Best-fit center wavelength [nm]
	fwhmBF: Best-fit FWHM [nm]
	ABF: Best-fit max count rate [1/s]
	'''
	if isinstance(p_bgd, (np.ndarray, np.generic) ):
		bgfxn 	= lambda x: np.polyval(p_bgd	,x)
	else:
		bgfxn 	= p_bgd	
	
	lam 	= uplobj.GetWavelength()
	ctrt 	= uplobj.GetCountRate()	
	if IDX is not None:
		ctrt 	= ctrt[:,IDX]
	def BestFitLorentzian(lam,ABF,lamBF,fwhmBF):		
		return np.abs(ABF) / (1.0 + np.power( (lam-lamBF)/(fwhmBF/2.0), 2)) + bgfxn(lam)

	#Pick the initial values
	ii 			= np.nonzero(np.logical_and( lam >= (lam0-0.5*NFWHM*fwhm0), lam <= (lam0+0.5*NFWHM*fwhm0)))[0]
	iinterp 	= spip.interp1d(lam,ctrt-bgfxn(lam))
	p0 			= np.polyfit(lam[ii],bgfxn(lam[ii]),1)	
	x0 			= np.array([iinterp(lam0),lam0,fwhm0])	
	def FitFxn(x):
		A 		= np.abs(x[0])
		x0 		= x[1]
		gamma 	= x[2] / 2.0
		p0 		= x[3]
		p1 		= x[4]
		return (ctrt[ii] - A / (1.0 + np.power( (lam[ii]-x0)/gamma, 2) ) - bgfxn(lam[ii]))

	#Do the fit
	xfit 	= np.abs(spop.leastsq(FitFxn, x0)[0])
	return Lorentzian(xfit[0], xfit[1], xfit[2], bgfxn)

def FitLorentzian3(uplobj, lam0, fwhm0, p_bgd, NFWHM=15, IDX=None):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	p_bgd:  Background level polynomial fit
	NFWHM: 	Number of FWHM on either side of lam0 to keep data
	Returns:
	lfxn: 	Best-fit Lorentzian function that takes an array of wavelengths [nm] as input
	lamBF: 	Best-fit center wavelength [nm]
	fwhmBF: Best-fit FWHM [nm]
	ABF: Best-fit max count rate [1/s]
	'''
	if isinstance(p_bgd, (np.ndarray, np.generic) ):
		bgfxn 	= lambda x: np.polyval(p_bgd	,x)
	else:
		bgfxn 	= p_bgd	
	
	lam 	= uplobj.GetWavelength()
	ctrt 	= uplobj.GetCounts()	
	if IDX is not None:
		ctrt 	= ctrt[:,IDX]
	def BestFitLorentzian(lam,ABF,lamBF,fwhmBF):		
		return np.abs(ABF) / (1.0 + np.power( (lam-lamBF)/(fwhmBF/2.0), 2)) + bgfxn(lam)

	#Pick the initial values
	ii 			= np.nonzero(np.logical_and( lam >= (lam0-0.5*NFWHM*fwhm0), lam <= (lam0+0.5*NFWHM*fwhm0)))[0]
	iinterp 	= spip.interp1d(lam,ctrt-bgfxn(lam))
	p0 			= np.polyfit(lam[ii],bgfxn(lam[ii]),1)	
	x0 			= np.array([iinterp(lam0),lam0,fwhm0,p0[0],p0[1]])	
	def FitFxn(x):
		A 		= np.abs(x[0])
		x0 		= x[1]
		gamma 	= x[2] / 2.0
		p0 		= x[3]
		p1 		= x[4]
		return (ctrt[ii] - A / (1.0 + np.power( (lam[ii]-x0)/gamma, 2) ) - np.polyval(np.array([p0,p1]),lam[ii]))

	#Do the fit
	xfit 	= spop.leastsq(FitFxn, x0)[0]
	bgfxn 	= lambda x: np.polyval(np.array([xfit[3],xfit[4]]),x)
	return Lorentzian(np.abs(xfit[0]), xfit[1], np.abs(xfit[2]), bgfxn)	

def FitGaussian(uplobj, lam0, fwhm0, NFWHM=15):
	'''
	Fit a Lorentzian-Cauchy distribution to the uPL object data
	PRECONDITION: Only one spectrum per file 
	Parameters:
	lam0: 	Initial resonant wavelength guess [nm] 
	fwhm0: 	Initial FWHM guess [nm]
	NFWHM: 	Number of FWHM on either side of lam0 to keep data
	Returns:
	Best-fit Gaussian function that takes an array of wavelengths [nm] as input
	'''
	lam 	= uplobj.GetWavelength()
	ctrt 	= uplobj.GetCountRate()
	ctrt0 	= ctrt.copy()	

	#for i in range(ctrt0.shape[1]):
	ctrt 	= ctrt0#[:,i]
	bg0 	= np.min(ctrt)#stats.mode(np.round(ctrt),axis=None)[0][0]
	ii 		= np.nonzero(np.logical_and( lam >= lam0-NFWHM*fwhm0, lam <= lam0+NFWHM*fwhm0 ))[0]	
	def FitFxn(x):
		A 		= np.abs(x[0])
		x0 		= x[1]
		sigma 	= x[2] / (2*np.sqrt(2*np.log(2)))
		bg 		= x[3] 
		#m 		= x[4]
		return (ctrt[ii] - A * np.exp(-np.power( (lam[ii]-x0) / (np.sqrt(2)*sigma),2)) - bg) / ii.shape[0]
	#Pick the initial values
	ii 		= np.nonzero(np.logical_and( lam >= (lam0-NFWHM*fwhm0), lam <= (lam0+NFWHM*fwhm0)))[0]
	lamii 	= lam[ii]
	#p1 		= np.polyfit(lamii,ctrt[ii],1)
	x0 		= np.array([np.max(ctrt[ii])-bg0,lamii[np.argmax(ctrt[ii])],fwhm0,bg0])
	#Do the fit
	xfit 	= spop.leastsq(FitFxn, x0)[0]
	#xfit,_,_ 	= spop.fmin_l_bfgs_b(FitFxn, x0, approx_grad=True, bounds=[(0,10*x0[0]),(x0[1]-1,x0[1]+1),(0.001,10*fwhm0),(0,bg0),(-1e4,1e4)])
	return [Gaussian(xfit[0], xfit[1], xfit[2]), xfit[3:]]
	
	
def EstimateFWHM(x,y,x0,pbkgd):	
	'''
	Estimate the full-width half maximum of a peak by moving from the estimated peak position <x0> to find where the signal is half-way to the background level 
	Parameters:
	x: x-data
	y: y-data 
	pbkgd: polynomial describing the background counts 
	'''
	if isinstance(pbkgd, (np.ndarray, np.generic) ):
		y0 		= y - np.polyval(pbkgd,x)
	else:
		y0 		= y - pbkgd(x)
	imax 	= np.argmin(np.abs(x-x0))
	y0max 	= y0[imax]
	#Find the left-hand side of the FWHM
	ilo 	= imax - 1 
	while ilo > 0 and y0[ilo] > 0.5*y0max:
		ilo 	= ilo - 1 
	if ilo < 0:
		ilo 	= 0
	#Find the right-hand side of the FWHM
	ihi 	= imax + 1 
	while ihi < x.shape[0]-1 and y0[ihi] > 0.5*y0max:
		ihi 	= ihi + 1 	
	if ihi >= x.shape[0]:
		ihi 	= x.shape[0]-1
	phi 	= np.polyfit(y0[(ihi-2):(ihi+2)],x[(ihi-2):(ihi+2)],1)
	plo 	= np.polyfit(y0[(ilo-2):(ilo+2)],x[(ilo-2):(ilo+2)],1)
	xlo 	= np.polyval(plo,0.5*y0max)
	xhi 	= np.polyval(phi,0.5*y0max)
	return np.abs(xhi-xlo)	
	
	
def IdentifyPeaks(uplobj,XCUT=None,exclWL=[435.8],qmin=200,qmax=None,IDX=None,POLYDEG=3,SIGCUT=2,NFWHM=2,SPIKEFILTER=None,DEBUG=True,SAVEFIG=False,FP=False):
	'''
	Identify peaks in the spectra, removing spurious peaks using low-pass filtering
	Parameters:
	uplobj: uPL object
	exclWL: wavelengths to exclude [e.g. due to laser line, Ne lights etc.]
	'''
	lam 	= uplobj.GetWavelength()
	y 		= uplobj.GetCounts()
	if XCUT is not None:
		y 	= y[lam > XCUT]
		lam 	= lam[lam > XCUT]
	if IDX is not None:
		y 	= y[:,IDX]
	assert(len(y.shape) == 1)
	grat 	= uplobj.GetGrating()
	if qmax == None:
		if grat == 1200:
			qmax 	= 6750 #Will be used to calculate cutoff frequencies for DSP
		elif grat == 1800:
			qmax 	= 10000 #Will be used to calculate cutoff frequencies for DSP
		elif grat == 2400:
			qmax 	= 8000 #Will be used to calculate cutoff frequencies for DSP
		else:
			print "Grating not supported for peak identification."
	if FP:
		#Filtering of Fabry-Perot oscillations
		
		#Resample over uniformly spaced grid 
		wl_us 	= np.linspace(np.min(lam),np.max(lam),lam.shape[0])
		i1d 	= spip.interp1d(lam,y)
		cts_us 	= i1d(wl_us)
		
		#Do the Fourier transform
		cts_fft 	= rfft(cts_us-np.mean(cts_us))
		wl_fft 		= rfftfreq(wl_us.shape[0],np.abs(wl_us[1]-wl_us[0]))		
		
		#Filter the signal
		sig 	= 6./3./2.
		f0_1 	= 15.2/2.  #14.9#
		f0_2 	= 17.4/2. #14.9#
		filt1	= 1 - np.exp(-np.power((wl_fft-f0_1)/(np.sqrt(2)*sig),2))
		filt2	= 1 - np.exp(-np.power((wl_fft-f0_2)/(np.sqrt(2)*sig),2))	
		filt 	= filt1*filt2
		
		if not SAVEFIG and DEBUG:
			plt.semilogy(wl_fft,np.abs(cts_fft),'k-',label="Signal")
			plt.semilogy(wl_fft,filt+1e-4,'r-',label="FP Filter")
			plt.legend(loc='upper right')
			plt.xlabel('Wavelength (nm)')
			plt.ylabel('Intensity (arb. units)')
			plt.show()
		
		y 	= irfft(cts_fft*filt) + np.mean(y)		
		lam = wl_us
		if not SAVEFIG and DEBUG:
			plt.plot(wl_us,cts_us,'k-',label="Signal")
			plt.plot(lam,y,'r-',label="FP Filter")
			plt.legend(loc='upper right')
			plt.grid()
			plt.xlabel('Wavelength (nm)')
			plt.ylabel('Intensity (arb. units)')
			plt.show()		
	
	if SPIKEFILTER is None:
		norm_pass 	= 0.3
		'''
		dlam 	= np.abs(np.mean(lam[1:] - lam[:-1]))
		lam0 	= np.mean(lam)
		fNyq 	= 0.5 / dlam #Nyquist frequency [1/nm]
		sig0 	= lam0 / (2*qmax*math.sqrt(2*math.log(2))) 
		#By Rayleigh criterion, the minimum peak-to-peak separation is sigma
		fCut 	= 0.5 / sig0
		'''
		#Get the normalized cutoff frequency
		norm_stop 		= 0.5
		#Apply a low-pass filter to get rid of spurious peaks
		(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
		(b, a) 	= signal.butter(N, Wn, btype='low', analog=0, output='ba')	
		yf = signal.lfilter(b, a, y)
	else:
		yf 	= signal.medfilt(y,SPIKEFILTER)
	
	def FitBackground(x,y,Nmax=100,TOL=0.005):
		#Fit the background with a polynomial iteratively eliminating outliers
		p_bgd 	= np.polyfit(x,y,POLYDEG)
		st0 	= np.std(y - np.polyval(p_bgd,x))		
		ii 		= np.nonzero( np.abs( y - np.polyval(p_bgd,x) ) <= 3*st0 )[0]
		Ni 		= 0 #Avoid infinite loops
		st1 	= 1e6
		while ii.shape[0] > 0 and Ni < Nmax and np.abs(1 - st1/st0) > TOL:
			#if DEBUG:
			#	print "N: %d std: %0.2e" % (Ni, st0)
			st1 	= st0 
			p_bgd 	= np.polyfit(x[ii],y[ii],POLYDEG)
			st0 	= np.std(y[ii] - np.polyval(p_bgd,x[ii]))		
			ii 		= np.nonzero( np.abs( y - np.polyval(p_bgd,x) ) <= 3*st0 )[0]			
			Ni 	= Ni + 1
		if Ni >= Nmax:
			#raise Exception("Background fit failed.  Inf looped")
			if DEBUG:
				print "Background fit failed.  Inf looped"
			return None 
		elif ii.shape[0] == 0:
			#raise Exception("Background fit failed.  ii sucked")
			if DEBUG:
				print "Background fit failed.  ii sucked"
			return None 
		else:
			return [p_bgd,st0]	
	try : 
		[p_bgd, st0] 	= FitBackground(lam,y)
	except :
		print "\t\tEXCEPTION FITTING BACKGROUND!"
		p_bgd 			= np.array([np.mean(y)])
		st0 			= np.std(y)
	ii				= signal.argrelextrema(yf, np.greater)[0]
	#only keep maxima with peaks more than SIGCUT ABOVE SIGNAL
	okind 			= np.nonzero((y[ii]-np.polyval(p_bgd,lam[ii])) > SIGCUT*st0)[0]
	if okind.shape[0] == 0:
		#raise Exception("No peaks found")
		if DEBUG:
			print "No peaks found"
		return None
	ii 				= ii[okind]
	
	#Identify peak FWHMs and fit 
	fwhm 			= np.zeros(ii.shape[0])
	fit1 			= []
	for i in range(ii.shape[0]):
		try:
			fwhm[i] 	= EstimateFWHM(lam,yf,lam[ii[i]],p_bgd)#lam[ii[i]]/1900.#
			fittmp 		= FitLorentzian3(uplobj, lam[ii[i]], fwhm[i], p_bgd, NFWHM=7, IDX=IDX)	
			#Store the noise level in the fit
			fittmp.SetNoise(st0)
			fittmp.SetFWHM0(fwhm[i])
			if DEBUG:
				print "Fit: %d\tLambda: %0.2f nm Q: %0.1f" % (i+1,fittmp.GetLambda0(),fittmp.GetLambda0()/fittmp.GetFWHM())
			qtmp 		= fittmp.GetLambda0()/fittmp.GetFWHM()
			if qtmp <= qmax and qtmp > qmin and not (np.round(exclWL,1) == np.round(fittmp.GetLambda0(),1)).any():
				fit1.append(fittmp)
			else:
				if DEBUG:
					print "Spurious peak detected at %0.1f nm" % (fittmp.GetLambda0())
		except:
			print "Fitting error"
	fit 			= []
	def TestEdge(fiti,NFWHM=0.5):
		lam0 	= fiti.GetLambda0()
		fwhm 	= fiti.GetFWHM()
		lam 	= uplobj.GetWavelength()
		return np.min(lam) < lam0 - NFWHM*fwhm and np.max(lam) > lam0 + NFWHM*fwhm
	#Now go through and make sure there are no duplicates
	for fiti in fit1:
		if fit is []:
			if TestEdge(fiti):
				fit.append(fiti)
			else:
				if DEBUG:			
					print "Failed edge test at lambda = %0.1f" % (fiti.GetLambda0())
		else:
			NODUP 	= False 
			for fitj in fit:
				dLam 	= np.abs(fiti.GetLambda0()-fitj.GetLambda0())
				sig 	= 0.5*np.max([fiti.GetFWHM(),fitj.GetFWHM()])
				if dLam < sig:
					NODUP = True 
					if DEBUG:
						print "Duplicate fit at lambda %0.1f" % (fiti.GetLambda0())
					break
			if not NODUP and TestEdge(fiti):
				fit.append(fiti)
			else:
				if DEBUG and not TestEdge(fiti):
					print "Failed edge test at lambda = %0.1f" % (fiti.GetLambda0())				
	if len(fit) == 0:
		print "FUCK YOU!"
		return None
	#Get rid of peaks at the edge of the measurement window 

	'''
	elif len(fit) > 1:
		if fit[0].GetLambda0() < fit[1].GetLambda0():
			fit 	= [fit[0]]
		else:
			fit 	= [fit[1]]	
	'''
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[12,12*2.0/(math.sqrt(5)+1)]}	 	
	matplotlib.rcParams.update(params)	
	plt.figure()
	if DEBUG:
		yinterp = spip.interp1d(lam,yf)
		plt.plot(lam,y,'k.',label='Raw',ms=4)
		plt.plot(lam,yf,'b-',label='Filtered')
		plt.plot(lam,np.polyval(p_bgd,lam),'r-',lw=2,label='Background')
		plt.plot(lam[ii],yf[ii],'y*',ms=8,label='Peaks')
		plt.plot(lam[ii]-0.5*fwhm,0.5*(yf[ii]+np.polyval(p_bgd,lam[ii])),'y^',ms=6)
		plt.plot(lam[ii]+0.5*fwhm,0.5*(yf[ii]+np.polyval(p_bgd,lam[ii])),'y^',ms=6,label='FWHM')
		maxes 	= []
		for fiti in fit:
			lami 	= np.linspace(fiti.GetLambda0()-0.5*fiti.GetFWHM()*NFWHM,fiti.GetLambda0()+0.5*fiti.GetFWHM()*NFWHM,1000)
			yi 		= fiti.GetFitFunction()(lami)
			maxes.append(np.max(yi))
			plt.plot(lami,yi,'g-')
		#Scale the figure properly 
		plt.ylim([10*np.round(0.8*np.min(np.polyval(p_bgd,lam))/10,0),10*np.round(1.2*np.max(maxes)/10,0)])
		plt.xlabel('Wavelength [nm]')
		plt.ylabel('Count rate [1/s]')
		plt.grid()
		#plt.legend(loc='upper right')
		if SAVEFIG:
			plt.savefig(SAVEFIG)
			plt.clf()
		else:
			plt.show()
			

	#Return the list of fit objects
	return fit
	
def IdentifyPeaksGauss(uplobj,exclWL=np.array([]),qmin=200,POLYDEG=3,IDX=None,SIGCUT=4,NFWHM=15,DEBUG=True,CONV=5,SAVEFIG=False):
	'''
	Identify peaks in the spectra, removing spurious peaks using low-pass filtering
	Parameters:
	uplobj: uPL object
	exclWL: wavelengths to exclude [e.g. due to laser line, Ne lights etc.]
	'''
	lam 	= uplobj.GetWavelength()
	y 		= uplobj.GetCountRate()
	if IDX is not None:
		y 	= y[:,IDX]
	assert(len(y.shape) == 1)
	grat 	= uplobj.GetGrating()
	if grat == 1200:
		qmax 	= 6000 #Will be used to calculate cutoff frequencies for DSP
		norm_pass 	= 0.1
	elif grat == 2400:
		qmax 	= 12000 #Will be used to calculate cutoff frequencies for DSP
		norm_pass 	= 0.1
	elif grat == 1800:
		qmax 	= 12000 #Will be used to calculate cutoff frequencies for DSP
		norm_pass 	= 0.1		
	else:
		print "Grating not supported for peak identification."

	norm_stop 		= 1.0
	#Apply a low-pass filter to get rid of spurious peaks
	(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=False)
	(b, a) 	= signal.butter(N, Wn, btype='low', analog=False, output='ba')	
	yf = signal.lfilter(b, a, y)

	def FitGaussian(x, y, xmin=250, xmax=2000, DEBUGFILE=False):
		'''
		Fit a single Gaussian peak plus a linear background to dataset (x,y), where x is in [eV] and y is counts
		
		PARAMETERS:
		x: wavelength vector [numpy vector]
		y: intensity vector [numpy vector]
		(Optional) xmin: low WL cut-off for fit [float/nm]
		(Optional) xmax: high WL cut-off for fit [float/nm]
		(Optional) DEBUGFILE: If this is not false, then raw data and fit will be plotted and saved to 
		the file name <DEBUGFILE>
		
		RETURNS:
		A list, where element #<x> is:
		0: The Gaussian function amplitude  [float/arb]
		1: The Gaussian function center [float/nm]
		2: The Gaussian full-width at half-maximum [float/nm]
		'''
		ii 		= np.nonzero(np.logical_and(x >= xmin, x <= xmax))[0]
		x 		= x[ii]
		y 		= y[ii]
		p_bgd 	= np.polyfit(x,y,POLYDEG)	
		ytst 	= np.polyval(p_bgd,x)
		a0 		= np.max(ytst)
		imx 	= np.argmax(ytst)
		x0 		= x[imx]
		ilo 	= imx - 1 
		ihi 	= imx + 1
		#Estimate the FWHM 
		while ilo > 1 and ytst[ilo] > 0.5*a0:
			ilo 	= ilo - 1
		while ihi < ii.shape[0] - 1 and ytst[ihi] > 0.5*a0:
			ihi 	= ihi + 1
		if ihi >= x.shape[0]:
			ihi 	= x.shape[0] - 1
		sig0 	= np.abs(x[ihi]-x[ilo]) / (2*np.sqrt(2*np.log(2)))
		#Estimate the initial background (form <y = m*x + b>)
		m 		= 0 
		b 		= np.min(ytst)
		#Define the function to minimize via the least-squares method 
		def ObjFxn(xf):
			A 		= np.abs(xf[0])
			x0 		= xf[1]
			sig 	= xf[2]
			m 		= xf[3]
			b 		= xf[4]
			if sig < 5:
				return 1e5*np.abs(y)
			else:
				return y - A*np.exp( -np.power((x-x0)/(math.sqrt(2)*sig), 2)) - m*x - b
		#Do the fit
		x0 		= np.array([a0,x0,sig0,m,b])
		xfit 	= spop.leastsq(ObjFxn, x0)[0]
		GaussObj 	= Gaussian(np.abs(xfit[0]), xfit[1], xfit[2]*2*math.sqrt(2*math.log(2)))
		def GaussFxn(x):
			A 		= np.abs(xfit[0]) 
			x0 		= xfit[1]
			sig 	= xfit[2]
			m 		= xfit[3]
			b 		= xfit[4]
			return A*np.exp( -np.power((x-x0)/(math.sqrt(2)*sig), 2)) + m*x + b
		
		#Return the fit parameters
		return [GaussObj,GaussFxn]

	#Fit the background and the curves until you have convergence
	[GaussObj,GaussBG] = FitGaussian(lam,yf)
	fit 					  = []
	yf0 					  = yf
	MYI 						  = 0 
	st0 					  = np.std(y - GaussBG(lam))
	while MYI < CONV :

		ii				= signal.argrelextrema(yf, np.greater)[0]	
		#only keep maxima with peaks more than SIGCUT ABOVE SIGNAL
		okind 			= np.nonzero(np.abs(y[ii]-GaussBG(lam[ii])) > SIGCUT*st0)[0]
		if okind.shape[0] == 0:
			#raise Exception("No peaks found")
			if DEBUG:
				print "No peaks found"
				break
		ii 				= ii[okind]
		
		#Identify peak FWHMs and fit 
		fwhm 			= np.zeros(ii.shape[0])
		fit1 			= []
		for i in range(ii.shape[0]):
			fwhm[i] 	= EstimateFWHM(lam,yf0,lam[ii[i]],GaussBG)
			fittmp 		= FitLorentzian2(uplobj, lam[ii[i]], fwhm[i], GaussBG, NFWHM=NFWHM, IDX=IDX)	
			#Store the noise level in the fit
			fittmp.SetNoise(st0)
			if DEBUG:
				print "Fit: %d\tLambda: %0.2f nm Q: %0.1f" % (i+1,fittmp.GetLambda0(),fittmp.GetLambda0()/fittmp.GetFWHM())
			qtmp 		= fittmp.GetLambda0()/fittmp.GetFWHM()
			if qtmp <= qmax and qtmp > qmin and not (np.round(exclWL,1) == np.round(fittmp.GetLambda0(),1)).any() :
				fit1.append(fittmp)
			else:
				if DEBUG:
					print "Spurious peak detected at %0.1f nm" % (fittmp.GetLambda0())
		fit 			= []
		#Now go through and make sure there are no duplicates
		for fiti in fit1:
			if fit is []:
				fit.append(fiti)
			else:
				NODUP 	= False 
				for fitj in fit:
					dLam 	= np.abs(fiti.GetLambda0()-fitj.GetLambda0())
					sig 	= 0.5*np.max([fiti.GetFWHM(),fitj.GetFWHM()])
					if dLam < sig:
						NODUP = True 
						if DEBUG:
							print "Duplicate fit at lambda %0.1f" % (fiti.GetLambda0())
						break
				if not NODUP:
					fit.append(fiti)
		#Revise the background by subtracting the Lorentzian peaks
		yf 				= yf0
		yi 				= y 		
		for fiti in fit:
			yf 			= yf - fiti.GetFitFunction()(lam) + GaussBG(lam)
			yi 			= yi - fiti.GetFitFunction()(lam) + GaussBG(lam)
		st0 					  = np.std(yi - GaussBG(lam))			
		[GaussObj,GaussBG] = FitGaussian(lam,yf)	
		MYI 				= MYI + 1

	
	if DEBUG:
		plt.plot(lam,y,'k.',label='Raw',ms=4)
		plt.plot(lam,yf0,'b-',label='Filtered')
		plt.plot(lam,GaussBG(lam),'r-',lw=2,label='Background')
		plt.plot(lam[ii],yf[ii],'y*',ms=8,label='Peaks')
		maxes 	= []
		for fiti in fit:
			lami 	= np.linspace(fiti.GetLambda0()-0.5*fiti.GetFWHM()*NFWHM,fiti.GetLambda0()+0.5*fiti.GetFWHM()*NFWHM,1000)
			yi 		= fiti.GetFitFunction()(lami)
			maxes.append(np.max(yi))
			plt.semilogy(lami,yi,'g-')
		if len(maxes) == 0:
			maxes 		= np.max(GaussBG(lam))
		else:
			plt.plot(lam[ii]-0.5*fwhm,0.5*(yf0[ii]+GaussBG(lam[ii])),'y^',ms=6)
			plt.plot(lam[ii]+0.5*fwhm,0.5*(yf0[ii]+GaussBG(lam[ii])),'y^',ms=6,label='FWHM')
		#Scale the figure properly 
		#plt.ylim([10*np.round(0.8*np.min(GaussBG(lam))/10,0),10*np.round(1.2*np.max(maxes)/10,0)])
		plt.xlabel('Wavelength [nm]')
		plt.ylabel('Count rate [1/s]')
		plt.grid()
		#plt.legend(loc='upper right')
		if SAVEFIG:
			plt.show()
			plt.savefig(SAVEFIG)
			plt.clf()
		else:
			plt.show()
	if len(fit) == 0:
		return [None,None]			
	#Return the list of fit objects
	return [fit,GaussObj]
	
if __name__=="__main__":

	font = {'family' : 'normal',
			'size'   : 16}
	
	matplotlib.rc('font', **font)
	
	test 		= LabSpec6("U:/Spectroscopy/QOLab/20160822-Testing/OSRAM_450nm_OD0_10um_1_0s_x30_50mA.txt","No cavity")
	i48_6sm 	= LabSpec6("U:/Spectroscopy/QOLab/20160823-Testing/OSRAM_450nm_ECAV_OD0_10um_0_1s_x60_48_6mA.txt","ECL")
	print "Grating: %d lp/mm" % (test.GetGrating())
	print "Integration time: %d s" % (test.GetIntegrationTime())
	print "Accumulations: %d" % (test.GetAccumulations())
	print "Center wavelength: %0.3f nm" % (test.GetCenterWavelength())	
	print "Slit width: %d microns" % (1000*test.GetSlitWidth())
	print test.GetTimestamp()
	plt.semilogy(test.GetWavelength(),test.GetCounts()/np.max(test.GetCounts()),'k-', label='As rec\'d / 50.0 mA')
	plt.semilogy(i48_6sm.GetWavelength(), i48_6sm.GetCounts()/np.max(i48_6sm.GetCounts()), 'b-', label='ECL / 48.6 mA')
	plt.xlabel("Wavelength [nm]")
	plt.ylabel("Normalized Intensity []")
	plt.legend(loc = 'upper right')
	plt.title('OSRAM 450 nm LD / ASE at Threshold')
	plt.grid()
	plt.ylim([1e-4,1.1])
	plt.show()
	'''
	up1 	= uPL("U:/Spectroscopy/uPL/20160216-A3405C/D4/E4_32_300uW_20s_50um_2400.dat",\
	 "1.6 kW/cm$^2$", 0.05, 20, 2400)
	#plt.plot(up1.GetWavelength(),up1.GetCountRate(),'k-',lw=1)
	#plt.show()
	IdentifyPeaks(up1,SAVEFIG="test.png")
	sys.exit()
	 
	lam 	= up1.GetWavelength()
	fit1 	= FitLorentzian(up1, 454, 0.08)
	fit2 	= FitLorentzian(up2, 454, 0.08)
	fit3 	= FitLorentzian(up3, 454, 0.08)
	fit4 	= FitLorentzian(up4, 454.8, 0.08)

	print "%s: lam %0.2f nm Q: %d A: %d" % (up1.GetTag(), fit1['lam'], \
	fit1['lam']/fit1['fwhm'], fit1['A'])
	print "%s: lam %0.2f nm Q: %d A: %d" % (up2.GetTag(), fit2['lam'], \
	fit2['lam']/fit2['fwhm'], fit2['A'])
	print "%s: lam %0.2f nm Q: %d A: %d" % (up3.GetTag(), fit3['lam'], \
	fit3['lam']/fit3['fwhm'], fit3['A'])
	print "%s: lam %0.2f nm Q: %d A: %d" % (up4.GetTag(), fit4['lam'], \
	fit4['lam']/fit4['fwhm'], fit4['A'])
	plt.plot(lam, up1.GetCountRate()-51, 'k.', lw=1, ms=4, label=up1.GetTag())
	plt.plot(lam, fit1['fit'](lam)-51, 'k-', lw=1)
	plt.plot(lam, up2.GetCountRate()-50, 'b.', lw=1, ms=4, label=up2.GetTag())
	plt.plot(lam, fit2['fit'](lam)-50, 'b-', lw=1)
	plt.plot(lam, up3.GetCountRate()-50, 'r.', lw=1, ms=4, label=up3.GetTag())
	plt.plot(lam, fit3['fit'](lam)-50, 'r-', lw=1)
	#plt.plot(lam, up4.GetCountRate(), 'm.', lw=1, ms=4, label=up4.GetTag())
	#plt.plot(lam, fit4['fit'](lam), 'm-', lw=1)
	plt.xlabel("Wavelength [nm]")
	plt.ylabel("Intensity [arb]")
	#plt.title('A3405B B4 (2,2)')
	plt.legend(loc='upper left')
	plt.xlim([fit1['lam']-1.5,fit1['lam']+1.5])#np.min(lam), np.max(lam)])
	plt.ylim([-1,51])
	plt.grid()
	plt.savefig('A3405B_B4_22.eps')
	plt.clf()
	plt.plot(lam, up1.GetCountRate()-fit1['fit'](lam), 'k.', lw=1, ms=4, label=up1.GetTag())
	plt.plot(lam, up2.GetCountRate()-fit2['fit'](lam), 'b.', lw=1, ms=4, label=up2.GetTag())
	plt.plot(lam, up3.GetCountRate()-fit3['fit'](lam), 'r.', lw=1, ms=4, label=up3.GetTag())
	plt.xlabel("Wavelength [nm]")
	plt.ylabel("Count rate [1/s]")
	plt.title('A3405B B4 (2,2) fit residuals')
	plt.legend(loc='upper left')
	plt.xlim([fit1['lam']-1.5,fit1['lam']+1.5])#np.min(lam), np.max(lam)])
	plt.grid()
	plt.savefig('A3405B_B4_22_resid.eps')
	#plt.show()	
	'''