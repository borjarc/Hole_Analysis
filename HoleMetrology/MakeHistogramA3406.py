import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpb
import scipy.stats as stat
from NanobeamHoles import * 
from SEMImage import *

class FitStatistics:

	def __init__(self,writeRadius):
		self.ebl 	= writeRadius 
		self.radii 	= []
		self.offcX 	= []
		self.offcY 	= []		
		
	def GetWriteRadius(self):
		return self.ebl 
		
	def GetDiameters(self):
		return np.array(self.radii)
		
	def GetOffsetsX(self):
		return np.array(self.offcX)

	def GetOffsetsY(self):
		return np.array(self.offcY)		
		
	def AddDiameters(self,rvec):
		self.radii.extend(rvec)
		
	def AddOffsetsX(self,offv):
		self.offcX.extend(offv)
		
	def AddOffsetsY(self,offv):
		self.offcY.extend(offv)		

def PlotHistogram(dat,lab=None,NBIN=7,color='k',off=0):
	bin0 		= np.mean(dat)
	binstd 		= np.std(dat)
	print "%0.1f +/- %0.1f nm N=%d" % (bin0,binstd,dat.shape[0])
	bin 		= np.linspace(bin0-2*binstd,bin0+2*binstd,NBIN)
	hist, bin 	= np.histogram(dat, bins=bin)
	width 		= 0.8 * (bin[1] - bin[0])
	ctr 		= 0.5*(bin[1:] + bin[:-1])
	if lab is not None:
		plt.bar(ctr+off*width, hist, align='center', width=width, color=color, label=lab)	
	else:
		plt.bar(ctr+off*width, hist, align='center', width=width, color=color)

if __name__=="__main__":
	font = {'family' : 'normal',
			'size'   : 18}

	mpb.rc('font', **font)	
	a3406 	= pickle.load(open('A3406.pkl','rb'))
	bot 		= {"1":[50,200],"2":[45,200],"4":[40,200],"5":[35,200],"6":[30,200],\
	"8":[50,230],"9":[45,230],"10":[40,230],"11":[35,230],"12":[30,230]}
	top 		= {"14":[50,200],"15":[45,200],"16":[40,200],"17":[35,200],"18":[30,200],\
	"20":[50,230],"21":[45,230],"22":[40,230],"23":[35,230],"24":[30,230]}	
	
	bDat 		= {50:FitStatistics(50), 45:FitStatistics(45), 40: FitStatistics(40), 35:FitStatistics(35), 30:FitStatistics(30)}
	tDat 		= {50:FitStatistics(50), 45:FitStatistics(45), 40: FitStatistics(40), 35:FitStatistics(35), 30:FitStatistics(30)}	
	
	for ky in a3406['top'].keys():
		writeR 			= top[ky][0]
		d 				= 2*a3406['top'][ky][-1].GetR()
		tDat[writeR].AddDiameters(d.tolist())
		if d.shape[0] > 3:
			x 			= a3406['top'][ky][-1].GetX0()
			y 			= a3406['top'][ky][-1].GetY0()
			p 			= np.polyfit(x,y,1)
			xmn 		= (x*(1+p[0])-p[0]*p[1])/(1+p[0]**2)
			ymn 		= np.polyval(p, xmn)
			poserr 		= y-ymn
			poserrX 	= x-xmn 
			tDat[writeR].AddOffsetsY(poserr.tolist())			
			tDat[writeR].AddOffsetsX(poserrX.tolist())				
	for ky in a3406['bot'].keys():
		writeR 			= bot[ky][0]	
		d 		= 2*a3406['bot'][ky][-1].GetR()
		bDat[writeR].AddDiameters(d.tolist())		
		if d.shape[0] > 3:
			x 			= a3406['bot'][ky][-1].GetX0()
			y 			= a3406['bot'][ky][-1].GetY0()
			p 			= np.polyfit(x,y,1)
			xmn 		= (x*(1+p[0])-p[0]*p[1])/(1+p[0]**2)
			ymn 		= np.polyval(p, xmn)
			poserr 		= y-ymn
			poserrX 	= x-xmn
			bDat[writeR].AddOffsetsY(poserr.tolist())		
			bDat[writeR].AddOffsetsX(poserrX.tolist())	
	i 		= 0 
	cols 	= ['k','b','r','g','m','c']
	for ky in bDat.keys():
		fobj 		= bDat[ky]
		fobj2 		= tDat[ky]
		dat 		= np.hstack([fobj.GetDiameters(),fobj2.GetDiameters()])
		PlotHistogram(dat,lab='$d_{EBL}$=%d'%(2*ky),color=cols[i%len(cols)])
		i 		= i + 1 
	plt.xlabel('Diameter [nm]')
	plt.ylabel('# Occurances')
	#plt.legend(loc='upper left')
	plt.savefig('20151118_a3406_diameter_stat.png')
	plt.clf()	
	dat 		= np.array([])
	for ky in bDat.keys():
		fobj 		= bDat[ky]
		fobj2 		= tDat[ky]
		dat 		= np.hstack([dat,fobj.GetOffsetsY(),fobj2.GetOffsetsY()])
		i 		= i + 1 
	print "\n"
	PlotHistogram(dat,color='k',NBIN=10)	
	plt.xlabel('Offset Y [nm]')
	plt.ylabel('# Occurances')
	#plt.legend(loc='upper left')
	plt.savefig('20151118_a3406_offY_stat.png')
	plt.clf()
	'''
	dat 		= np.array([])
	for ky in bDat.keys():
		fobj 		= bDat[ky]
		fobj2 		= tDat[ky]
		dat 		= np.hstack([dat,fobj.GetOffsetsX(),fobj2.GetOffsetsX()])
		i 		= i + 1 
	print "\n"
	PlotHistogram(dat,color='k',NBIN=10)	
	plt.savefig('20151118_a3406_crap_stat.png')	
	plt.xlabel('Offset X [nm]')
	plt.ylabel('# Occurances')
	#plt.legend(loc='upper left')
	plt.show()		
	'''
		