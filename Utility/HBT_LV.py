import numpy as np 
import csv
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import signal
import numpy.linalg as linalg

class HBT_LV:

	def __init__(self,filen,t0):
		'''
		Holds information on HBT exported LabView csv files 
		'''
		assert(os.path.exists(filen))	
		self.t0 				= t0 
		[self.tau,self.cts] 	= self.__LoadFile__(filen,delim="\t")
		
	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 	= 0
			tau	= []
			cts = []		
			for row in spamreader:
				if row[0] is not []:
					tau.append(float(row[0]))
					cts.append(float(row[1]))
			tau 		= np.array(tau)
			cts 		= np.array(cts)
			return [tau, cts]
			
	def GetT0(self):
		return self.t0 
		
	def SetT0(self,t0):
		self.t0 	= t0 
		
	def GetTau(self):
		return self.tau - self.t0
		
	def GetCounts(self):
		return self.cts 
		
	def GetNormalizedCounts(self,Wn=0.05,DEBUG=False):
		#Get the normalized cutoff frequency
		norm_pass 		= 0.1	
		norm_stop 		= 0.4
		#Apply a low-pass filter to get rid of spurious peaks
		#(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
		(b, a) 	= signal.butter(4, Wn, btype='low', analog=False, output='ba')	
		yf 		= signal.filtfilt(b, a, self.cts)	
		tau 	= self.tau - self.t0 
		a 		= np.polyfit(tau,yf,1)
		if DEBUG:
			plt.plot(self.tau-self.t0, self.cts,'k-')
			plt.plot(self.tau-self.t0, yf,'b-')	
			plt.plot(self.tau-self.t0,np.polyval(a,tau),'r--')
			plt.xlabel('$\\tau$ (ns)')
			plt.ylabel('$g^{(2)}$')
			plt.grid()
			plt.show()
		return self.cts/a[1]

if __name__=="__main__":
	hbt 	= HBT_LV('U:/Analysis/20170329-g2/Small_325nm_150bin.csv', 8.0)
	hbt.GetNormalizedCounts()