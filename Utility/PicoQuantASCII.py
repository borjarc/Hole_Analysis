import numpy as np
import scipy.interpolate as spip
import scipy.optimize as spop
import matplotlib as mpb
import matplotlib.pyplot as plt
import math
import os.path 
import sys
import csv
from scipy import stats
from scipy import signal

class PicoQuantASCII:

	def __init__(self, filen, delim="\t"):
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
		[self.nCh, self.tCh, self.tVec, self.cVec] 	= self.__LoadFile__(filen,delim)

	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			spamreader.next()
			spamreader.next()
			nCh		= int(spamreader.next()[0])
			spamreader.next()
			spamreader.next()			
			spamreader.next()		
			spamreader.next()			
			spamreader.next()				
			tCh		= float(spamreader.next()[0])
			tVec	= tCh * np.arange(nCh)
			cVec 	= np.zeros(nCh)
			i 		= 0
			for row in spamreader:
				try:
					cVec[i] 	= int(spamreader.next()[0])
					i 	= i + 1
				except:
					continue
		return [nCh, tCh, tVec, cVec]
	
	def GetNchannel(self):
		return self.nCh 
		
	def GetTchannel(self):
		return self.tCh 
		
	def GetTime(self):
		return self.tVec
		
	def GetCounts(self):
		return self.cVec
		
	def GetT0(self):
		return self.tVec[np.argmax(self.cVec)]
		
	def GetMaxCts(self):
		return np.max(self.cVec)
			
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[0.75*12,12*2.0/(math.sqrt(5)+1)]}	
	mpb.rcParams.update(params)	
	ref 	= PicoQuantASCII("U:/Spectroscopy/QOLab/20161230-A3531-Ctr/266 nm on sample amplifier full_16ps-TRPL.dat")
	plt.plot(ref.GetTime()-ref.GetT0(),ref.GetCounts()/ref.GetMaxCts(),'k-',lw=1.5,label='Ref')
	gan 	= PicoQuantASCII("U:/Spectroscopy/QOLab/20161230-A3531-Ctr/Sq_266nm_496uW_OD2_54_GaN-TRPL.dat")
	plt.plot(gan.GetTime()-ref.GetT0(),gan.GetCounts()/gan.GetMaxCts(),'b-',lw=1.5,label='GaN')
	gan 	= PicoQuantASCII("U:/Spectroscopy/QOLab/20161230-A3531-Ctr/Sq_266nm_496uW_OD2_54_QW-TRPL.dat")
	plt.semilogy(gan.GetTime()-ref.GetT0(),gan.GetCounts()/gan.GetMaxCts(),'r-',lw=1.5,label='QW')	
	plt.xlim([-1,2])
	plt.xlabel('Time [ns]')
	plt.ylabel('Counts []')
	plt.grid()
	plt.legend(loc='upper right')
	plt.show()
