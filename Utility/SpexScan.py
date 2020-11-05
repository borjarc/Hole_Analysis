import numpy as np
import matplotlib as mpb
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import csv

class SpexScan:

	def __init__(self,wl,sw,tag,filen,Nx,Ny):
		self.wl 	= wl 
		self.sw 	= sw 
		self.tag 	= tag 
		self.filen 	= filen
		self.Nx 	= Nx 
		self.Ny 	= Ny 
		[self.x,self.y,self.cts] 	= self.__ParseFile__(filen)
		[self.xf,self.yf,self.gridz] 		= self.__CalculateRegularGrid__(self.x,self.y,self.cts)
		
	def __ParseFile__(self,filen,delim=','):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			j 			= 0
			x 			= []
			y 			= []
			cts 		= []
			row 		= spamreader.next()
			while not (row[0] == "X_Value"):
				row 	= spamreader.next()
			for row in spamreader:
				if len(row) == 4:
					x.append(float(row[1]))
					y.append(float(row[2]))
					cts.append(float(row[3]))
			x 			= np.array(x)
			y 			= np.array(y)
			cts 		= np.array(cts)
		return [x,y,cts]	
		
	def GetTag(self):
		return self.tag
		
	def GetSW(self):
		return self.sw 
		
	def GetFilename(self):
		return self.filen 
		
	def GetWavelength(self):
		return self.wl 
		
	def GetX(self):
		return self.x 
		
	def GetY(self):
		return self.y 
		
	def GetNx(self):
		return self.Nx 
		
	def GetNy(self):
		return self.Ny 
		
	def GetIntensity(self):
		return self.cts
		
	def GetXr(self):
		return self.xf 
		
	def GetYr(self):
		return self.yf 
		
	def GetIntensityr(self):
		return self.gridz
		
	def SetTag(self,tag):
		self.tag 	= tag 
		
	def SetSW(self,sw):
		self.sw 	= sw 
		
	def SetFile(self, filen):
		self.filen 		= filen 
		[self.x,self.y,self.cts] 	= self.__ParseFile__(filen)
	
	def SetWavelength(self,wl):
		self.wl 		= wl
		
	def SetNx(self,Nx):
		self.Nx 		= Nx 
		
	def SetNy(self,Ny):
		self.Ny 		= Ny 
	
	def __CalculateRegularGrid__(self,x,y,z):
		#Interpolate the scan values over a normal grid
		xf 			= np.linspace(np.min(x),np.max(x),self.Nx )
		yf 			= np.linspace(np.min(y),np.max(y),self.Ny )
		[xg,yg] 	= np.meshgrid(xf,yf)
		gridz 		= spi.griddata((x,y),z,(xg.flatten(),yg.flatten()),method='linear')
		return [xf,yf,gridz]
		
	def MakePlot(self):
		plt.clf()
		cs 	= plt.contourf(self.xf,self.yf,self.gridz.reshape(self.yf.shape[0],self.xf.shape[0]))
		plt.xlim([np.min(self.xf),np.max(self.xf)])		
		plt.ylim([np.min(self.yf),np.max(self.yf)])
		plt.axis('equal')
		plt.xlabel('x [$\mu$m]')
		plt.ylabel('y [$\mu$m]')
		plt.colorbar(cs)
		#plt.clabel('Count rate [1/s]')
		
if __name__=="__main__":
	pass