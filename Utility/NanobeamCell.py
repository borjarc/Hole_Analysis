import numpy as np 
from uPL import *
import matplotlib
import matplotlib.pyplot as plt
from NanobeamUPL import * 
from scipy.cluster.vq import *

class NanobeamCell:

	def __init__(self,cID,a,rb,rc):
		self.cID 	= cID 
		self.a 		= a 
		self.rb 	= rb 
		self.rc 	= rc
		self.qB 	= []
		self.qR 	= []
		self.aB 	= []
		self.aR 	= []
		self.lB		= []
		self.lR 	= []
		self.split 	= []
		
	def GetCellID(self):
		return self.cID
		
	def SetCellID(self,cID):
		cID 	= self.cID
				
	def GetA(self):
		return self.a 
		
	def GetRb(self):
		return self.rb 
		
	def GetRc(self):
		return self.rc 
		
	def SetA(self,a):
		self.a 	= a 

	def SetRc(self,rc):
		self.rc 	= rc 
		
	def SetRb(self,rb):
		self.rb 	= rb
	
	def AddQB(self, obj):
		self.qB.append(obj)
		
	def AddQR(self, obj):
		self.qR.append(obj)	
		
	def AddAB(self, obj):
		self.aB.append(obj)
		
	def AddAR(self, obj):
		self.aR.append(obj)		
		
	def AddLB(self, obj):
		self.lB.append(obj)
		
	def AddLR(self, obj):
		self.lR.append(obj)
		
	def AddSplit(self, obj):
		self.split.append(obj)
		
	def GetQB(self):
		return np.array(self.qB)[0]
		
	def GetQR(self):
		return np.array(self.qR)[0]
		
	def GetAB(self):
		return np.array(self.aB)[0]
		
	def GetAR(self):
		return np.array(self.aR)[0]
		
	def GetLB(self):
		return np.array(self.lB)[0]
		
	def GetLR(self):
		return np.array(self.lR)[0]
		
	def GetSplit(self):
		return np.array(self.split)[0]
		
	def GetMeanQB(self):
		return np.mean(self.GetQB())	
		
	def GetMeanQR(self):
		return np.mean(self.GetQR())
					
	def GetMeanAB(self):
		return np.mean(self.GetAB())
		
	def GetMeanAR(self):
		return np.mean(self.GetAR())	
		
	def GetMeanLB(self):
		return np.mean(self.GetLB())
		
	def GetMeanLR(self):
		return np.mean(self.GetLR())	
		
	def GetMeanSplit(self):
		return np.mean(self.GetSplit())
		
class NanobeamCellCluster:

	def __init__(self,cID,a,rb,rc,Ncluster,amplitude,lam,q,LOG=False):
		self.cID 		= cID 
		self.a 			= a 
		self.rb 		= rb 
		self.rc 		= rc
		self.q 			= []
		self.amp 		= []
		self.lam 		= []
		self.Ncluster 	= Ncluster
		self.q 			= q
		self.amp	 	= amplitude
		self.lam 		= lam
		self.LOG 		= LOG
		self.indLst 	= self.__InitializeCluster__(lam,q)
		self.split 		= []
		
	def GetCellID(self):
		return self.cID
		
	def SetCellID(self,cID):
		cID 	= self.cID
				
	def GetA(self):
		return self.a 
		
	def GetRb(self):
		return self.rb 
		
	def GetRc(self):
		return self.rc 
		
	def GetNcluster(self):
		return self.Ncluster
		
	def SetNcluster(self,Ncluster):	
		self.Ncluster 	= Ncluster
		
	def SetA(self,a):
		self.a 	= a 

	def SetRc(self,rc):
		self.rc 	= rc 
		
	def SetRb(self,rb):
		self.rb 	= rb
		
	def SetLog(self,LOG):
		self.LOG 	= LOG 
		
	def GetLog(self):
		return self.LOG
		
	def GetSplit(self):
		return np.array(self.split)[0]
		
	def AddSplit(self, obj):
		self.split.append(obj)		
		
	def __RecalculateCluster__(self,lam,q,cInd):
		'''
		Fracture the Q/lambda data into Ncluster clusters.  Return the indeces corresponding to
		cluster <cInd>, which is sorted in terms of wavelength
		Preconditions:
		lam and q are np.ndarray vectors of same length
		cInd is an integer greater than = zero or less than Ncluster
		'''
		assert(lam.shape[0] == q.shape[0])
		assert(len(lam.shape) == 1)
		assert(cInd >= 0 and cInd < self.Ncluster)
		#Do the k-means clustering per the SciPy reference
		if self.LOG:
			dat 		= np.vstack([lam,np.log(q)]).transpose() #Each row of dat is an observation
		else:
			dat 		= lam.transpose()
		ctrd, lab 	= kmeans2(dat, self.Ncluster)
		#Compute the average wavelength for each cluster
		inds 		= np.arange(lam.shape[0])
		lbar 		= np.zeros(self.Ncluster)
		for i in range(self.Ncluster):
			if np.any(lab == i):
				lbar[i] 	= np.mean(lam[inds[lab == i]])
			else:
				return None
		lasort 		= np.argsort(lbar)
		return inds[lab == lasort[cInd]]
		
	def __InitializeCluster__(self,lam,q):
		'''
		Fracture the Q/lambda data into Ncluster clusters.  Return the indeces corresponding to
		cluster <cInd>, which is sorted in terms of wavelength
		Preconditions:
		lam is an np.ndarray vector
		'''
		assert(len(lam.shape) == 1)
		#Do the k-means clustering per the SciPy reference
		if self.LOG:
			dat 		= np.vstack([lam,np.log(q)]).transpose() #Each row of dat is an observation
		else:
			dat 		= lam.transpose()
		ctrd, lab 	= kmeans2(dat, self.Ncluster)
		#Compute the average wavelength for each cluster
		inds 		= np.arange(lam.shape[0])
		lbar 		= np.zeros(self.Ncluster)
		for i in range(self.Ncluster):
			if np.any(lab == i):
				lbar[i] 	= np.mean(lam[inds[lab == i]])
			else:
				return None
		lasort 		= np.argsort(lbar)
		indLst 		= []
		for cInd in range(self.Ncluster):
			indLst.append(inds[lab == lasort[cInd]])
		return indLst
		
	def PlotCluster(self,cInd,col='ko'):
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.__RecalculateCluster__(lam,q,cInd)
		if ii is not None:
			plt.plot(lam[ii],q[ii],col,ms=8)
		
	def GetQ(self, ind):
		assert(ind < self.Ncluster)
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		if ii is not None and ii.shape[0] > 0:
			return q[ii]
		else:
			return None
		
	def GetAmplitude(self, ind):
		assert(ind < self.Ncluster)	
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		amp 	= np.array(self.amp)
		if ii is not None and ii.shape[0] > 0:		
			return amp[ii]
		else:
			return None			
		
	def GetLambda(self, ind):
		assert(ind < self.Ncluster)
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		if ii is not None and ii.shape[0] > 0:		
			return lam[ii]
		else:
			return None			
		
	def GetMeanQ(self,ind):
		assert(ind < self.Ncluster)
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		if ii is not None and ii.shape[0] > 0:
			return np.mean(q[ii])
		else:
			return None			
		
	def GetMeanAmplitude(self,ind):
		assert(ind < self.Ncluster)	
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		amp 	= np.array(self.amp)
		if ii is not None and ii.shape[0] > 0:		
			return np.mean(amp[ii])
		else:
			return None			
					
	def GetMeanLambda(self, ind):
		assert(ind < self.Ncluster)
		q  		= np.array(self.q)
		lam 	= np.array(self.lam)
		ii 		= self.indLst[ind]
		if ii is not None and ii.shape[0] > 0:		
			return np.mean(lam[ii])
		else:
			return None			