import numpy as np
import scipy.interpolate as spip
import scipy.optimize as spop
import matplotlib
import matplotlib.pyplot as plt
import math
import os.path 
import sys
import csv
from scipy import stats
from scipy import signal
from datetime import datetime
import copy
from uPL import *
import datetime

class macroPL(uPL):
	def __init__(self, filen, intT, sw, grat, lam0, Navg, tag, delim="\t"):
		'''
		Create an object representing a spectrum recorded using the LabSpec6
		Parameters:
		filen: 		full filename
		intT: 		integration time (s)
		sw: 		slit width (um)
		grat: 		grating (lp/mm)
		lam0: 		center wavelength (nm)
		Navg: 		number of averages (int)
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))
		[self.lam, self.spectra, self.timestamp] 	= self.__LoadFile__(filen,delim)
		self.intT 	= intT
		self.sw		= sw
		self.grat	= grat 
		self.lam0	= lam0
		self.N 		= Navg
		self.tag 	= tag
		
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []	
			row 		= spamreader.next()
			assert("0001:AREA1:1-Channel(X)" in row[0])
			for row in spamreader:
				if len(row) == 2:
					wl.append(float(row[0]))
					cts.append(float(row[1]))
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			timestamp 	= datetime.datetime.fromtimestamp( os.path.getmtime(filen) )
			return [wl, cts, timestamp]	

class PLbuddy(uPL):

	def __init__(self, filen, tag, delim="\t"):
		'''
		Create an object representing a spectrum recorded using the LabSpec6
		Parameters:
		filen: 		full filename
		intT: 		integration time (s)
		sw: 		slit width (um)
		grat: 		grating (lp/mm)
		lam0: 		center wavelength (nm)
		Navg: 		number of averages (int)
		tag: 		string to help remember the file
		'''
		assert(os.path.exists(filen))	
		[self.lam, self.spectra, self.timestamp, self.grat, self.sw, self.intT, self.temp, self.pwr] 	= self.__LoadFile__(filen,delim)
		self.lam0	= self.lam
		self.N 		= 1
		self.tag 	= tag

	def GetTemperature(self):
		return self.temp 
		
	def GetPower(self):
		return self.pwr

	def __ParseFilename__(self,filen,i=0):
		delim	= "_"
		filen 	= filen.split("/")[-1]
		
		eles 	= filen.split(delim)
		temp 	= int(eles[i+1].split("K")[0])
		pwr 	= int(eles[i+2].split("uW")[0])
		intT 	= float(eles[i+3].split("s")[0])
		grat 	= int(eles[i+7][1:])
		sw 		= eles[i+5].split("mm")[0]
		sw 		= int(sw.split(".")[1])
		return [grat, sw, intT, temp, pwr]		

	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			wl 			= []
			cts 		= []	
			row 		= spamreader.next()
			assert("Wavelength" in row[0])
			for row in spamreader:
				if len(row) == 2:
					wl.append(float(row[0]))
					cts.append(float(row[1]))
			wl 			= np.array(wl)
			cts 		= np.array(cts)
			timestamp 	= datetime.datetime.fromtimestamp( os.path.getmtime(filen) )
			[grat, sw, intT, temp, pwr] 	= self.__ParseFilename__(filen)
			return [wl, cts, timestamp, grat, sw, intT, temp, pwr]			

if __name__=="__main__":
	filen 	=	 "U:/Spectroscopy/PL/2017-11-13-A3572/TempSeries/TempSeriesLowPower/A3572_0p25mW_900g_100um_13K_60s.dat"
	mPL 	= macroPL(filen, 60., 100., 900, 445, 1, "13 K")
	plt.plot(mPL.GetWavelength(),mPL.GetCountRate())
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Count rate (1/s)')
	plt.grid()
	plt.show()