from macroPL import *
from uPL import *
import numpy as np
import matplotlib as mpb 
import math 
import matplotlib.pyplot as plt 
import numpy as np
import cPickle as pickle
import sys
import numpy as np
from uPL import *
import os
from SpectrumUtilities import *
from os import listdir
from os.path import isfile, join
from scipy.optimize import brenth, leastsq
#from LoadMacroPL import *

h 		= 6.626e-34 #[eV-s]
hbar 	= h / (2*math.pi)
c 		= 3e8 #[m/s]
nm2cm 	= 1e-7
kB 		= 1.381e-23 #[J/K]
m0 		= 9.11e-31 #[kg]
eV2J 	= 1.609e-19

def ParseFile(f):
	'''
	A custom function designed to read TU-Berlin's data of center energy versus temperature
	
	PARAMTERS:
	f: file name [string]
	
	RETURNS:
	A list, where element #<x> is:
	0: Temperature [numpy vector/K]
	1: The peak energy [numpy vector/eV]
	'''
	T 	= []
	eV 	= []
	with open(f, 'rb') as csvfile:
		rd = csv.reader(csvfile, delimiter=',')
		for row in rd:
			T.append(float(row[0]))
			eV.append(float(row[1]))
		eV 		= np.array(eV)
		T 		= np.array(T)
		return [T, eV]	

def SolveX(xv,Ti):	
	#Only works for Ti scalar
	El 		= 0.001 * xv[1] * eV2J #meV->J
	sigl 	= 0.001 * xv[2] * eV2J #meV->J
	gRatio 	= xv[3]
	def xfun(x):
		return x*np.exp(x) - gRatio*( (sigl/(kB*Ti))**2 - x)*np.exp(El/(kB*Ti))
	try:
		x0 	= brenth(xfun, 0, (sigl/(kB*Ti))**2)
	except:
		return 0

	return x0
	
def ModVarshni(xv,T):
	Eg0 		= xv[0]
	alpha_V 	= 1e-3*0.835 #eV/K 
	beta_V 		= 774. #K
	xvec 		= np.zeros(T.shape[0])
	for i in range(T.shape[0]):
		xvec[i] 	= SolveX(xv,T[i])
	return Eg0 - xvec*8.6173e-5*T - alpha_V*np.power(T,2)/(T+beta_V) 
	
def FitFunction(xv,T,Eg):
	#Call this function to do the weighted fit
	return ModVarshni(xv,T)-Eg
	
if __name__=="__main__":
	E0 		= 2.98 + 0.05  
	Eloc 	= 89. #[meV]
	sigL 	= 29. #[meV]
	gRatio 	= 20.
	x0 		= np.array([E0,Eloc,sigL,gRatio])
	
	
	[T325,E325,I325,FW325] = LoadMacro325(TMAX=160.)
	[Te, Ee] 	= ParseFile("Irene_Epl.csv")
	
	xfit 		= spop.least_squares(lambda x: FitFunction(x, T325, E325),x0).x
	print "2 nm A3530:"
	print xfit
	Tth 	= np.linspace(4,300,100)
	Eth 	= ModVarshni(xfit,Tth)
	
	plt.errorbar(T325,E325,0.0033*np.ones(E325.shape[0]),fmt='ko',ms=10)
	plt.plot(Tth,Eth,'k--',lw=2)

	xfit 		= spop.least_squares(lambda x: FitFunction(x, Te, Ee),x0).x
	print "3 nm A2916:"
	print xfit	
	Eth 	= ModVarshni(xfit,Tth)	
	plt.errorbar(Te,Ee,0.0033*np.ones(Te.shape[0]),fmt='ro',ms=10)
	plt.plot(Tth,Eth,'r--',lw=2)	
	
	plt.ylabel('$E_{PL}$ (eV)',color='k')
	plt.xlabel('$T$ (K)')

	plt.grid()	
	plt.show()