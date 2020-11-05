import numpy as np 
from uPL import *
import matplotlib
import matplotlib.pyplot as plt

class NanobeamUPL:

	def __init__(self, sampID, cellID, row, col):
		self.sID 		= sampID 
		self.cID 		= cellID
		self.row 		= row 
		self.col 		= col 
		self.uPL 		= {}	
	
	def GetSampleID(self):
		return self.sID 
	
	def GetCellID(self):
		return self.cID  
		
	def GetRow(self):
		return self.row 
		
	def GetColumn(self):
		return self.col 
		
	def SetSampleID(self,sID):
		self.sID  	= sID 
	
	def SetCellID(self, cID):
		self.cID  	= cID  
		
	def SetRow(self, r):
		self.row 	= r	
		
	def SetColumn(self,c):
		self.col = c 
		
	def AddUPL(self,label,fitobj):
		self.uPL[label] 	= {'fit':fitobj}
	'''	
	def GetUPL(self,label):
		if self.uPL.has_key(label):
			return self.uPL[label]['uPL']
		else:
			return None 
	'''		
	def GetFit(self,label):
		if self.uPL.has_key(label):
			return self.uPL[label]['fit']
		else:
			return None 
	'''		
	def GetAllUPL(self):
		return map(lambda x: self.uPL[x]['uPL'], self.uPL.keys())
	'''
	
	def GetAllKeys(self):
		return self.uPL.keys()
		
	def GetAllFits(self):
		dat 	= map(lambda x: self.uPL[x]['fit'], self.uPL.keys())
		return dat[0]
	
	def GetAll(self):
		return self.uPL
		
class NanobeamPolUPL(NanobeamUPL):

	def __init__(self, sampID, cellID, row, col,pol):
		NanobeamUPL.__init__(self, sampID, cellID, row, col)
		self.pol 	= pol 
		
	def GetPolarization(self):
		return self.pol 
		
	def SetPolarization(self,pol):
		self.pol 	= pol 
		
class NanobeamPwrUPL(NanobeamUPL):

	def __init__(self, sampID, cellID, row, col, pwr):
		NanobeamUPL.__init__(self, sampID, cellID, row, col)
		self.pwr 	= pwr 
		
	def GetPower(self):
		return self.pwr 
		
	def SetPower(self,pwr):
		self.pwr 	= pwr 		