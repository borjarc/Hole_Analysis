import cv2
import numpy as np
import math
import matplotlib as mpb
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.interpolate import interp2d
from scipy.optimize import least_squares
from scipy.integrate import trapz
import csv
import os.path as osp
import sys

class Field2D:

	def __init__(self,xc,yc,zc):
		'''
		Parent class for vector and scalar fields.  zc is assumed to be vertical coordinate, just a label really.
		<z> is surface normal.
		'''
		assert(len(xc.shape) == 1)
		assert(len(yc.shape) == 1) 
		
		self.x 		= xc 
		self.y 		= yc 
		self.z 		= zc 
		
	def GetX(self):
		return self.x 
		
	def GetY(self):
		return self.y 
		
	def GetZ(self):
		return self.z
		
	def GetKx(self):
		dx 	= np.mean(np.abs(self.x[:-1]-self.x[1:]))
		return 2*math.pi*fft.fftshift(fft.fftfreq(np.max(self.x.shape),d=dx))
		
	def GetKy(self):
		dy 	= np.mean(np.abs(self.y[:-1]-self.y[1:]))
		return 2*math.pi*fft.fftshift(fft.fftfreq(np.max(self.y.shape),d=dy))		
		
class ScalarField2D(Field2D):

	def __init__(self,xc,yc,zc,f,INTERP=False):
		Field2D.__init__(self,xc,yc,zc)
		self.f 		= f 
		if INTERP:
			self.myitp 	= interp2d(xc,yc,f)
		else:
			self.myitp 	= None
		
	def GetFFT(self):
		return fft.fftshift(fft.fft2(self.f))
			
	def GetF(self):
		return self.f
	
	def F(self,xc,yc): #Evaluate the field
		return self.myitp(xc,yc)
		
	def Integrate2D(self):
		return trapz(trapz(self.f,x=self.x.flatten()),x=self.y.flatten())
		
	def PlotField(self,Nlev=16,mcm=plt.cm.jet):
		levels 	= np.linspace(np.min(np.min(self.f)),np.max(np.max(self.f)),Nlev)
		plt.contourf(self.x,self.y,self.f,levels,cmap=mcm)
		
class RoughSurface(ScalarField2D):
	'''
	Class to hold information about a rough surface.
	'''
	def __init__(self,xc,yc,zc,f,RMS,dz,INTERP=False):
		#Zero mean 
		f 		= f.astype('float32')
		f 		= f - np.mean(f)
		#Give the surface unit RMS surface roughness
		rmsNorm = np.sqrt(np.sum(np.sum(np.power(f,2))) / (f.shape[0]*f.shape[1]))
		f 		= f / rmsNorm
		ScalarField2D.__init__(self,xc,yc,zc,f,INTERP=INTERP)
		self.fmin 	= np.min(np.min(f))
		self.fmax	= np.max(np.max(f))
		self.rms 	= RMS #nanometer
		self.dz 	= dz  #nanometer
		self.N 		= int(np.ceil( (self.fmax-self.fmin)*self.rms / self.dz))
		self.dz 	= (self.fmax-self.fmin)*self.rms/self.N
		self.levels = np.linspace(self.fmin,self.fmax,self.N+2)[1:-1]
		
	def GetDz(self):
		return self.dz
		
	def GetRMS(self):
		return self.rms 
		
	def GetNslc(self):
		#Compute the number of slices
		return self.N
		
	def PlotImage(self,scale=1.,SAVEFIG=None,EQ=False):
		rsmat 			= self.f
		rsmat 			= rsmat - np.min(np.min(rsmat))
		rsmat 			= rsmat / np.max(np.max(rsmat))
		img 			= rsmat*255
		img 			= np.round(img).astype('uint8')
		img 			= cv2.resize(img,(0,0),fx=scale,fy=scale)
		if EQ:
			img 		= cv2.equalizeHist(img)
		if SAVEFIG:
			cv2.imwrite(SAVEFIG,img)
		else:
			cv2.imshow('image',img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()			
		
	#def PlotField(self,Nlev=16,mcm=plt.cm.jet):
		#plt.contourf(self.x,)
		#f 		= self.f + np.min(np.min(self.f))
		#f 		= np.round(255 * f / np.max(np.max(f)))
		#f 		= f.astype('uint8')
		#cv2.imshow('image',f)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	def GetSlice(self,i):
		#Return a boolean array of height values; 1 to change refractive index, 0 otherwise 
		assert(i >= 0 and i < self.N)
		return (self.f <= self.levels[i])
		
	def VisualizeStack(self):
		for i in range(self.GetNslc()):
			print "\t Slice %d/%d" % (i+1,self.GetNslc())
			slc 	= self.GetSlice(i).astype('uint8')*255
			cv2.imshow('image',cv2.resize(slc,(slc.shape[1]*2,slc.shape[0]*2)))
			cv2.waitKey(0)
			cv2.destroyAllWindows()	
	
class VectorField2D(Field2D):

	def __init__(self,xc,yc,zc,fx,fy,fz,INTERP=True):
		Field2D.__init__(self,xc,yc,zc)
		self.fx 	= fx 
		self.fy 	= fy 
		self.fz 	= fz
		if INTERP:
			self.Fx 	= interp2d(xc,yc,fx)
			self.Fy 	= interp2d(xc,yc,fy)
			self.Fz 	= interp2d(xc,yc,fz)
		else:
			self.Fx 	= None 
			self.Fy 	= None 
			self.Fz 	= None
		
	def GetFx(self):
		return  self.fx 
		
	def GetFy(self):
		return  self.fy 

	def GetFz(self):
		return  self.fz 		
		
	def Fx(self,xc,yc): #Evaluate the x-field
		if (len(xc.shape) == 1 and len(yc.shape) == 1):
			return self.Fx(xc,yc)
		else:
			return self.__Interpolate__(xc,yc,self.Fx)
		
	def Fy(self,xc,yc): #Evaluate the y-field
		if (len(xc.shape) == 1 and len(yc.shape) == 1):
			return self.Fy(xc,yc)
		else:
			return self.__Interpolate__(xc,yc,self.Fy)
		
	def Fz(self,xc,yc): #Evaluate the z-field
		if (len(xc.shape) == 1 and len(yc.shape) == 1):
			return self.Fz(xc,yc)
		else:
			return self.__Interpolate__(xc,yc,self.Fz)
		
	def __Interpolate__(self,xc,yc,fxn):
		#Have to resort to this because interp2d is written stupidly
		ans 		= np.zeros(xc.shape)
		for i in range(xc.shape[0]):
			for j in range(xc.shape[1]):
				ans[i,j] 	= fxn(xc[i,j],yc[i,j])
		return ans
		
	def __GetFFT__(self,arr,FILTER=True):
		#Apply a top-hat filter to avoid aliasing
		if FILTER:
			x 		= np.linspace(-1,1,arr.shape[0])
			y 		= np.linspace(-1,1,arr.shape[1])
			[yg,xg] 	= np.meshgrid(y,x)
			filt 	= np.exp(-np.power( xg/0.80, 10)-np.power(yg/0.80, 10))
		else:
			filt 	= np.ones(arr.shape[0]).astype('float32')
		return fft.fftshift(fft.fft2(filt*arr,norm='ortho'))/np.sqrt(arr.shape[0]*arr.shape[1])
		
	def GetFFTx(self):
		return self.__GetFFT__(self.fx)
	
	def GetFFTy(self):
		return self.__GetFFT__(self.fy) 
	
	def GetFFTz(self):
		return self.__GetFFT__(self.fz)
		
	def __PlotContour__(self,fxn,Nlev=16,mcm=plt.cm.jet):
		levels 	= np.linspace(np.min(np.min(fxn)),np.max(np.max(fxn)),Nlev)
		plt.contourf(self.x,self.y,fxn,levels,cmap=mcm)	
		#plt.gca().set_aspect('equal')
		
	def PlotFieldX(self,Nlev=16,mcm=plt.cm.jet):
		self.__PlotContour__(np.abs(self.fx),Nlev,mcm)

	def PlotFieldY(self,Nlev=16,mcm=plt.cm.jet):
		self.__PlotContour__(np.abs(self.fy),Nlev,mcm)	
		
	def PlotFieldZ(self,Nlev=16,mcm=plt.cm.jet):
		self.__PlotContour__(np.abs(self.fz),Nlev,mcm)

	def __CalcIntensity__(self):
		return np.power(np.abs(self.fx),2) + np.power(np.abs(self.fy),2) + np.power(np.abs(self.fz),2)
		
	def GetIntensity():
		return self.__CalcIntensity__()
	
	def PlotIntensity(self,Nlev=16,mcm=plt.cm.jet):
		intens 	= self.__CalcIntensity__()
		self.__PlotContour__(intens,Nlev,mcm)	

	def PlotAmplitude(self,Nlev=16,mcm=plt.cm.jet):
		ampl 	= np.sqrt(self.__CalcIntensity__())
		self.__PlotContour__(ampl,Nlev,mcm)				
		
def LoadScalarField(filen,LAB2="y(microns)("):
	if osp.exists(filen):
		with open(filen, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			row 		= spamreader.next()
			#Load in x-coordinates
			ux 		= len(row[0].split("x(microns)("))					
			while spamreader and ux <= 1:
				row 		= spamreader.next()
				if len(row) > 0:
					ux 		= len(row[0].split("x(microns)("))
				else:
					ux 		= 1
			Nx 	= int(row[0].split("x(microns)(")[1].split(",")[0])
			ux 	= np.zeros(Nx)
			for i in range(Nx):
				row 		= spamreader.next()
				ux[i] 		= float(row[1])
				
			row 		= spamreader.next()
			row 		= spamreader.next()
				
			Ny 	= int(row[0].split(LAB2)[1].split(",")[0])
			uy 	= np.zeros(Ny)
			for i in range(Ny):
				row 		= spamreader.next()
				uy[i] 		= float(row[1])
			#Read in the data
			Pout 		= np.zeros((Ny,Nx),dtype='float64')
			row 			= spamreader.next()
			row 			= spamreader.next()
			for i in range(Nx):
				row 		= spamreader.next()
				for j in range(1,Ny+1):
					Pout[j-1,i] 	= float(row[j])
			return [ux,uy,Pout]
			
	else:
		raise Exception('File %s not found.' % filen) 
		
def CalcPoynting(ef,hf):
	'''
	Calculate the Poynting vector from two vector field objects 
	'''
	eX 		= ef.GetFx()
	eY 		= ef.GetFy()
	hX 		= np.conj(hf.GetFx())
	hY 		= np.conj(hf.GetFy())
	assert(eX.shape[0] == hY.shape[0])
	assert(eX.shape[1] == hY.shape[1])
	assert(hX.shape[0] == eY.shape[0])
	assert(hX.shape[1] == eY.shape[1])	
	assert(eX.shape[0] == eY.shape[0])
	assert(eX.shape[1] == eY.shape[1])
	arg 	= eX*hY - eY*hX
	P 		= -0.5*np.real(arg).astype('float32') #Poynting vector is real
	return ScalarField2D(ef.GetX(),ef.GetY(),ef.GetZ(),P )
	
def FTVectorField(vf,INTERP=False):
	kx 	= vf.GetKx()
	ky 	= vf.GetKy()
	fx 	= vf.GetFFTx() 
	fy 	= vf.GetFFTy()
	fz 	= vf.GetFFTz()
	zc 	= vf.GetZ()
	return VectorField2D(kx,ky,zc,fx,fy,fz,INTERP=INTERP)

def CreateScalarFieldLumerical(filen,zc,LAB2="y(microns)("):
	[xc,yc,f] 	= LoadScalarField(filen,LAB2=LAB2)
	return ScalarField2D(xc*1e-6,yc*1e-6,zc,f,INTERP=True)
	
def CreateVectorFieldLumerical(fileX,fileXa,fileY,fileYa,fileZ,fileZa,zc,LAB2="y(microns)(",FLIPY=False):
	[xc,yc,fx] 	= LoadScalarField(fileX,LAB2=LAB2)
	[_,_,fy] 	= LoadScalarField(fileY,LAB2=LAB2)
	[_,_,fz] 	= LoadScalarField(fileZ,LAB2=LAB2)
	[_,_,fxa] 	= LoadScalarField(fileXa,LAB2=LAB2)
	[_,_,fya] 	= LoadScalarField(fileYa,LAB2=LAB2)
	[_,_,fza] 	= LoadScalarField(fileZa,LAB2=LAB2)	
	if FLIPY:
		mul 	= -1 
	else:
		mul 	= 1
	return VectorField2D(xc*1e-6,mul*yc*1e-6,zc,fx*np.exp(1j*fxa),fy*np.exp(1j*fya),fz*np.exp(1j*fza)) #Convert um to SI units (m).
	
def LoadTE00plusE(zc=6e-9):
	return CreateVectorFieldLumerical("TE00/Ex_abs_top_above.txt","TE00/Ex_phs_top_above.txt",\
	"TE00/Ey_abs_top_above.txt","TE00/Ey_phs_top_above.txt","TE00/Ez_abs_top_above.txt",\
	"TE00/Ez_phs_top_above.txt",zc)
	
def LoadTE00plusH(zc=6e-9):
	return CreateVectorFieldLumerical("TE00/Hx_abs_top_above.txt","TE00/Hx_phs_top_above.txt",\
	"TE00/Hy_abs_top_above.txt","TE00/Hy_phs_top_above.txt","TE00/Hz_abs_top_above.txt",\
	"TE00/Hz_phs_top_above.txt",zc)	
	
def LoadTE10plusE(zc=6e-9):
	return CreateVectorFieldLumerical("TE10/Ex_abs_top_above.txt","TE10/Ex_phs_top_above.txt",\
	"TE10/Ey_abs_top_above.txt","TE10/Ey_phs_top_above.txt","TE10/Ez_abs_top_above.txt",\
	"TE10/Ez_phs_top_above.txt",zc)
	
def LoadTE10plusH(zc=6e-9):
	return CreateVectorFieldLumerical("TE10/Hx_abs_top_above.txt","TE10/Hx_phs_top_above.txt",\
	"TE10/Hy_abs_top_above.txt","TE10/Hy_phs_top_above.txt","TE10/Hz_abs_top_above.txt",\
	"TE10/Hz_phs_top_above.txt",zc)		
	
def LoadTE00xzplusE(zc=6e-9):
	return CreateVectorFieldLumerical("TE00xz/Ex_abs_top_above.txt","TE00xz/Ex_phs_top_above.txt",\
	"TE00xz/Ez_abs_top_above.txt","TE00xz/Ez_phs_top_above.txt","TE00xz/Ey_abs_top_above.txt",\
	"TE00xz/Ey_phs_top_above.txt",zc,LAB2="z(microns)(")
	
def LoadTE00xzplusH(zc=6e-9):
	return CreateVectorFieldLumerical("TE00xz/Hx_abs_top_above.txt","TE00xz/Hx_phs_top_above.txt",\
	"TE00xz/Hz_abs_top_above.txt","TE00xz/Hz_phs_top_above.txt","TE00xz/Hy_abs_top_above.txt",\
	"TE00xz/Hy_phs_top_above.txt",zc,LAB2="z(microns)(")	
	
def LoadTE10xzplusE(zc=6e-9):
	return CreateVectorFieldLumerical("TE10xz/Ex_abs_top_above.txt","TE10xz/Ex_phs_top_above.txt",\
	"TE10xz/Ez_abs_top_above.txt","TE10xz/Ez_phs_top_above.txt","TE10xz/Ey_abs_top_above.txt",\
	"TE10xz/Ey_phs_top_above.txt",zc,LAB2="z(microns)(")
	
def LoadTE10xzplusH(zc=6e-9):
	return CreateVectorFieldLumerical("TE10xz/Hx_abs_top_above.txt","TE10xz/Hx_phs_top_above.txt",\
	"TE10xz/Hz_abs_top_above.txt","TE10xz/Hz_phs_top_above.txt","TE10xz/Hy_abs_top_above.txt",\
	"TE10xz/Hy_phs_top_above.txt",zc,LAB2="z(microns)(")	

def sLoadTE00plusE(zc=6e-9):
	return CreateVectorFieldLumerical("sTE00/Ex_abs_top_above.txt","sTE00/Ex_phs_top_above.txt",\
	"sTE00/Ey_abs_top_above.txt","sTE00/Ey_phs_top_above.txt","sTE00/Ez_abs_top_above.txt",\
	"sTE00/Ez_phs_top_above.txt",zc)
	
def sLoadTE00plusH(zc=6e-9):
	return CreateVectorFieldLumerical("sTE00/Hx_abs_top_above.txt","sTE00/Hx_phs_top_above.txt",\
	"sTE00/Hy_abs_top_above.txt","sTE00/Hy_phs_top_above.txt","sTE00/Hz_abs_top_above.txt",\
	"sTE00/Hz_phs_top_above.txt",zc)	

def sLoadTE10plusE(zc=6e-9):
	return CreateVectorFieldLumerical("sTE10/Ex_abs_top_above.txt","sTE10/Ex_phs_top_above.txt",\
	"sTE10/Ey_abs_top_above.txt","sTE10/Ey_phs_top_above.txt","sTE10/Ez_abs_top_above.txt",\
	"sTE10/Ez_phs_top_above.txt",zc)
	
def sLoadTE10plusH(zc=6e-9):
	return CreateVectorFieldLumerical("sTE10/Hx_abs_top_above.txt","sTE10/Hx_phs_top_above.txt",\
	"sTE10/Hy_abs_top_above.txt","sTE10/Hy_phs_top_above.txt","sTE10/Hz_abs_top_above.txt",\
	"sTE10/Hz_phs_top_above.txt",zc)		
	
def LoadEpsilon00(zc=0):
	return CreateScalarFieldLumerical("eps00.txt",zc)
	
def LoadEpsilon10(zc=0):
	return CreateScalarFieldLumerical("eps10.txt",zc)	
	
def sLoadEpsilon00(zc=0):
	return CreateScalarFieldLumerical("seps00.txt",zc)
	
def sLoadEpsilon10(zc=0):
	return CreateScalarFieldLumerical("seps10.txt",zc)		
	
def LoadEpsilon00xz(zc=0):
	return CreateScalarFieldLumerical("eps00xz.txt",zc,LAB2="z(microns)(")
	
def LoadEpsilon10xz(zc=0):
	return CreateScalarFieldLumerical("eps10xz.txt",zc,LAB2="z(microns)(")	
	
def LoadSurfaceFromFile(filen,rms,dz,dx=6.8187e-6,dy=200e-9,zc=0):
	img 	= cv2.imread(filen,0)
	xc 		= np.linspace(-0.5*dx,0.5*dx,img.shape[1])
	yc 		= np.linspace(-0.5*dy,0.5*dy,img.shape[0])
	return RoughSurface(xc,yc,zc,img,rms,dz,INTERP=True)
	
def TestRoughSurface():
	x 			= np.linspace(-10,10,600)
	y 			= np.linspace(-1,1,10)
	[xg,yg] 	= np.meshgrid(x,y)
	f 			= np.cos(xg*2*math.pi/3.) #Period 2 
	rs 			= RoughSurface(x,y,0,f,1,0.5)
	print "Slices: %d" % (rs.GetNslc())
	rs.PlotField()
	plt.show()
	for i in range(rs.GetNslc()):
		slc 	= rs.GetSlice(i)
		lev 	= np.array([0,0.25,0.5,0.75,1])
		plt.contourf(x,y,slc,lev,axis='equal')
		plt.show()
	
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 30,
	'figure.autolayout':True, 'figure.figsize':[12,12]}	
	mpb.rcParams.update(params)	
	#TestRoughSurface()

	plt.gca().set_aspect('equal')
	e00plus 	= LoadTE10plusE()
	h00plus 	= LoadTE10plusH()	
	#eK00plus 	= FTVectorField(e00plus)
	#eK00plus.PlotAmplitude()
	#h00plus 	= LoadTE00plusH()
	#e00plus.PlotIntensity()
	p00 		= CalcPoynting(e00plus,h00plus)
	p00.PlotField()
	print p00.Integrate2D()
	plt.show()