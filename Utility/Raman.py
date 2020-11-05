import numpy as np
import math
import os.path 
import sys
import csv
from uPL import *
import matplotlib as mpb
import matplotlib.pyplot as plt

class Raman(uPL):

	def __init__(self, filen, tag, sw, intT, grat, laserWL, delim="\t"):
		'''
		Create an object representing spectra recorded using the LabView program on
		the uPL setup.
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		sw: 		slit width [mm]
		intT:		integration time [s]
		laserWL: 	laser wavelength [nm]
		grat: 		grating ruling [lp/mm]
		'''
		assert(os.path.exists(filen))
		self.tag 	= tag
		self.sw 	= sw 
		self.intT 	= intT 
		self.laser  = laserWL
		self.grat 	= grat 
		[self.lam, self.spectra] 	= self.__LoadFile__(filen,delim)
		
	def GetLaserWavelength(self):
		return self.laser 
		
	def SetLaserWavelength(self, WL):
		self.laser 	= laserWL
		
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 		= 0
			wl 		= []
			cts 	= []		
			for row in spamreader:
				#Skip the first row
				if j == 0:
					j 	= j + 1
					continue
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
			
	def GetWavenumber(self):
		return self.lam
		
class RamanSweep(Raman):

	def __init__(self, filen, tag, sw, intT, grat, laserWL, delim="\t"):
		'''
		Create an object representing spectra recorded using the LabView program on
		the uPL setup.
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		sw: 		slit width [mm]
		intT:		integration time [s]
		laserWL: 	laser wavelength [nm]
		grat: 		grating ruling [lp/mm]
		'''
		assert(os.path.exists(filen))
		self.tag 	= tag
		self.sw 	= sw 
		self.intT 	= intT 
		self.laser  = laserWL
		self.grat 	= grat 
		[self.height, self.lam, self.spectra] 	= self.__LoadFile__(filen,delim)
		
	def GetWavenumber(self, h):
		ii 	= np.nonzero(h == self.height)[0]
		return self.lam[ii]
		
	def GetCounts(self,h):
		ii 	= np.nonzero(h == self.height)[0]
		return self.spectra[ii]		
		
	def GetCountRate(self,h):
		ii 	= np.nonzero(h == self.height)[0]
		return self.spectra[ii]	/ self.intT

	def GetThirdAxis(self):
		return np.sort(np.unique(self.height))
		
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 		= 0
			height 	= []
			wl 		= []
			cts 	= []		
			for row in spamreader:
				#Skip the first row
				if j == 0:
					j 	= j + 1
					continue
				height.append(float(row[0]))
				wl.append(float(row[1]))
				subCts 	= []
				for i in range(2,len(row)):
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
			height 		= np.array(height)
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			if cts.shape[1] == 1:
				cts 	= cts[:,0]
			return [height, wl, cts]		
			
class RamanMapping(Raman):

	def __init__(self, filen, tag, sw, intT, grat, laserWL, delim="\t"):
		'''
		Create an object representing spectra recorded using the LabView program on
		the uPL setup.
		Parameters:
		filen: 		full filename
		tag: 		string to help remember the file
		sw: 		slit width [mm]
		intT:		integration time [s]
		laserWL: 	laser wavelength [nm]
		grat: 		grating ruling [lp/mm]
		'''
		assert(os.path.exists(filen))
		self.tag 	= tag
		self.sw 	= sw 
		self.intT 	= intT 
		self.laser  = laserWL
		self.grat 	= grat 
		self.spectra = np.array([])
		self.lam 	= np.array([])
		[self.x, self.y, self.wn, self.mapping] 	= self.__LoadFile__(filen,delim)
		self.fits 	= None
		
	def GetWavenumberMapping(self, x, y):
		ii 	= np.nonzero(np.logical_and(x == self.x, y == self.y))[0]
		return self.wn[ii]
		
	def GetCountsMapping(self, x, y):
		ii 	= np.nonzero(np.logical_and(x == self.x, y == self.y))[0]
		return self.mapping[ii]		
		
	def GetCountRateMapping(self, x, y):
		ii 	= np.nonzero(np.logical_and(x == self.x, y == self.y))[0]
		return self.mapping[ii]	/ self.intT

	def GetX(self):
		return np.sort(np.unique(self.x))
		
	def GetY(self):
		return np.sort(np.unique(self.y))
		
	def DoFits(self,DEBUG=True):
		'''
		Fit the spectra for each point in the mapscan individually.
		Store the fit objects as a list of lists
		'''
		fits 			= []
		xvec 			= self.GetX()
		yvec 			= self.GetY()
		[xg,yg] 		= np.meshgrid(xvec,yvec)
		xvec 			= xg.flatten()
		yvec 			= yg.flatten()
		for i in range(xvec.shape[0]):
			xi 			= xvec[i]
			yi 			= yvec[i]
			#A bit hacky, but the uPL fitting algorithm will use self.lam and self.spectra for fitting
			self.lam	 = self.GetWavenumberMapping(xi,yi)
			self.spectra = self.GetCountsMapping(xi,yi)
			fits.append(IdentifyPeaks(self,exclWL=[],qmin=5,qmax=1000,POLYDEG=1,SIGCUT=1,XCUT=60,NFWHM=3,DEBUG=DEBUG))#,SAVEFIG=DEBUG))
			fitsEnd 	 = fits[-1]
			lam0 		 = np.array(map(lambda x: x.GetLambda0(), fitsEnd))
			fwhm 		 = np.array(map(lambda x: x.GetFWHM(), fitsEnd))
			print "x: %0.1f y: %0.1f" % (xi,yi)
			for j in range(lam0.shape[0]):
				print "\t nu: %0.1f fwhm: %0.2f" % (lam0[j],fwhm[j])
		self.xvec 		= xvec 
		self.yvec 		= yvec
		
	def GetFits(self,wn,TOL=5.):
		#Get all fits corresponding to wavenumbers of
		if fits is not None:
			pass
		
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 		= 0
			x 		= []
			y 		= []
			wl 		= []
			cts 	= []		
			for row in spamreader:
				#Skip the first row
				if j == 0:
					j 	= j + 1
					continue
				x.append(float(row[0]))
				y.append(float(row[1]))
				wl.append(float(row[2]))
				subCts 	= []
				for i in range(3,len(row)):
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
			x 			= np.array(x)
			y 			= np.array(y)
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			if cts.shape[1] == 1:
				cts 	= cts[:,0]
			return [x, y, wl, cts]				
		
if __name__=="__main__":
	filen 	= "U:/Spectroscopy/Raman/2016-05-11 Formation/A2916_green_488_100xL.txt"
	filen2 	= "U:/Spectroscopy/Raman/2016-05-11 Formation/A2916_green_405_100xL1.txt"	
	filen3 	= "U:/Spectroscopy/Raman/2016-05-11 Formation/A3405A-B2-SiRef_405.txt"
	myR 	= Raman(filen, "A2916 Green", 0.02, 1, 2400, 488)
	myR2 	= Raman(filen2, "A2916 Green", 0.02, 1, 2400, 405)	
	myR3 	= Raman(filen3, "A3405A Si", 0.02, 1, 2400, 405)	
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'text.fontsize' : 28,
	'figure.autolayout':True, 'figure.figsize':[12,12*2.0/(math.sqrt(5)+1)]}	
	mpb.rcParams.update(params)		
	plt.semilogy(myR.GetWavenumber(),myR.GetCountRate()/np.max(myR.GetCountRate()),'k-',ms=4, label='$\lambda=488$ nm')
	plt.semilogy(myR2.GetWavenumber(),myR2.GetCountRate()/np.max(myR2.GetCountRate()),'b-',ms=4, label='$\lambda=405$ nm')	
	plt.semilogy(myR3.GetWavenumber(),myR3.GetCountRate()/np.max(myR3.GetCountRate()),'r-',ms=4, label='Si $\lambda=405$ nm')	
	plt.grid()
	#plt.xlim([100,900])
	plt.legend(loc='lower right')
	plt.xlabel('Wavenumber [cm$^{-1}$]')
	plt.ylabel('Counts [arb]')
	plt.show()