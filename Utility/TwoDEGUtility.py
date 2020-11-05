import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
from scipy.stats import linregress
from scipy.optimize import leastsq
import matplotlib as mpb
from math import pi, sqrt
from scipy.interpolate import interp1d

class PVTrace:

	'''
	Object to read and hold photovoltage data 
	'''

	def __init__(self, filen, tag, delim=','):
		self.tag 	= tag 
		self.filen 	= filen
		[self.t,self.Rs,self.V] 	= self.__LoadFile__(filen, delim)

	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			t 		= []
			Rs	 	= []
			V 		= [] 	
			row 	= spamreader.next()
			try:
				while row and not("Channel" in row[0]):
					row 	= spamreader.next()
				for row in spamreader:
					t.append(float(row[0]))					
					Rs.append(float(row[1]))
					V.append(float(row[2]))
			except:
				return [None, None, None]
			t 		= np.array(t)
			Rs 		= np.array(Rs)
			V		= np.array(V)
			return [t, Rs, V]
			
	def GetRs(self):
		return self.Rs 
		
	def GetT(self):
		return self.t 
		
	def GetV(self):
		return self.V 
		
	def GetTag(self):
		return self.tag 
		
	def GetFilename(self):
		return self.filen
		
	def SetFile(self, filen):
		self.filen 	= filen
		[self.t,self.Rs,self.V] 	= self.__LoadFile__(filen, delim)
		
	def GetEdgesOn(self):
		#Returns the indeces where the photodiode is turned on
		return (np.nonzero(np.logical_and(self.V[:-1] == 0, self.V[1:] > 0))[0] + 1)
		
	def GetEdgesOff(self):
		#Returns the indeces where the photodiode is turned off
		return np.hstack([np.array(0),np.nonzero(np.logical_and(self.V[:-1] > 0, self.V[1:] == 0))[0]])
	
	def GetEdges(self):
		#Returns the indeces where the photodiode is turned on or off
		return np.sort(np.setunion1d(self.GetEdgesOn,self.GetEdgesOff))
		
	def GetNumberPeriods(self):
		return self.GetEdgesOn().shape[0]
		
	def GetNthOn(self,N):
		#Return the data corresponding to the Nth "on" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		elif N == (self.GetNumberPeriods() - 1):
			onI 	= self.GetEdgesOn()
			return [ self.t[onI[-1]:], self.Rs[onI[-1]:], self.V[onI[-1]:] ]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			return [ self.t[onI[N]:offI[N+1]], self.Rs[onI[N]:offI[N+1]], \
			self.V[onI[N]:offI[N+1]] ]
			
	def GetNthOff(self,N):
		#Return the data corresponding to the Nth "off" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		elif N == 0:
			onI 	= self.GetEdgesOn()
			return [ self.t[:onI[0]], self.Rs[:onI[0]], self.V[:onI[0]] ]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			return [ self.t[(offI[N]+1):onI[N]], self.Rs[(offI[N]+1):onI[N]], \
			self.V[(offI[N]+1):onI[N]] ]	
			
	
class PVTrace2(PVTrace):	
	'''
	Object to read and hold photovoltage versus time data for starting with LED on and 
	ending with it off.
	'''
	def __init__(self, filen, tag, off_txCut, off_tlCut, on_txCut, on_tlCut, delim=','):
		PVTrace.__init__(self,filen,tag,delim)
		self.pwr = self.V		
		N 		= self.GetNumberPeriods()

		onFit 	= []
		offFit	= []	
		
		for i in range(N):
			[ti,ri,vi] 	= self.GetNthOn(i)
			if ti is not None and ti.shape[0] > 10 and np.max(ti-ti[0]) > on_tlCut:
				onFit.append(SingleTraceFit(ti,ri,on_txCut,on_tlCut))
				#onFit[-1].PlotFits('b','on %d' % (i))				
			[ti,ri,vi] 	= self.GetNthOff(i)
			if ti is not None and ti.shape[0] > 10 and np.max(ti-ti[0]) > off_tlCut:
				offFit.append(SingleTraceFit(ti,ri,off_txCut,off_tlCut))
				#offFit[-1].PlotFits('k','off %d' % (i))
		self.onFit 		= onFit 
		self.offFit 	= offFit

	#Accessor methods for fit objects
	def GetOnFits(self):
		return self.onFit 
		
	def GetOffFits(self):
		return self.offFit 
		
	def GetTauxOn(self):
		taux 	= np.zeros(len(self.onFit))
		for i in range(len(self.onFit)):
			taux[i] 	= self.onFit[i].GetTaux() #* self.onFit[i].GetAx()
		return taux
		
	def GetTauxOff(self):
		taux 	= np.zeros(len(self.offFit))
		for i in range(len(self.offFit)):
			taux[i] 	= self.offFit[i].GetTaux() #* self.offFit[i].GetAx()
		return taux

	def GetAxOn(self):
		Ax 	= np.zeros(len(self.onFit))
		for i in range(len(self.onFit)):
			Ax[i] 	= self.onFit[i].GetAx() #* self.onFit[i].GetAx()
		return Ax	
		
	def GetAxOff(self):
		Ax 	= np.zeros(len(self.offFit))
		for i in range(len(self.offFit)):
			Ax[i] 	= self.offFit[i].GetAx() #* self.offFit[i].GetAx()
		return Ax	

	def GetBxOn(self):
		Bx 	= np.zeros(len(self.onFit))
		for i in range(len(self.onFit)):
			Bx[i] 	= self.onFit[i].GetBx() #* self.onFit[i].GetBx()
		return Bx	
		
	def GetBxOff(self):
		Bx 	= np.zeros(len(self.offFit))
		for i in range(len(self.offFit)):
			Bx[i] 	= self.offFit[i].GetBx() #* self.offFit[i].GetAx()
		return Bx	
		
	def GetTaulOn(self):
		taul 	= np.zeros(len(self.onFit))
		for i in range(len(self.onFit)):
			taul[i] 	= self.onFit[i].GetTaul() #* self.offFit[i].GetAx()
		return taul	
		
	def GetTaulOff(self):
		taul 	= np.zeros(len(self.offFit))
		for i in range(len(self.offFit)):
			taul[i] 	= self.offFit[i].GetTaul() #* self.offFit[i].GetAx()
		return taul	

	def GetTaulErrOn(self):
		return np.array(lambda x: x.GetTaulerr(), self.onFit)
		
	def GetTaulErrOff(self):
		return np.array(lambda x: x.GetTaulerr(), self.offFit)	

	def GetAlOn(self):
		return np.array(lambda x: x.GetAl(), self.onFit)
		
	def GetAlOff(self):
		return np.array(lambda x: x.GetAl(), self.offFit)	

	def GetAlErrOn(self):
		return np.array(lambda x: x.GetAlerr(), self.onFit)
		
	def GetAlErrOff(self):
		return np.array(lambda x: x.GetAlerr(), self.offFit)		
	
	def GetEdgesOn(self):
		#Returns the indeces where the photodiode is turned on
		return np.hstack([0,np.nonzero(np.logical_and(self.pwr[:-1] < 0.1, self.pwr[1:] > 0.1))[0]])
		
	def GetEdgesOff(self):
		#Returns the indeces where the photodiode is turned off
		return np.nonzero(np.logical_and(self.pwr[:-1] > 0.1, self.pwr[1:] < 0.1))[0]
		
	def GetNthOn(self,N):
		#Return the data corresponding to the Nth "on" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			try:
				return [ self.t[onI[N]:offI[N]], self.Rs[onI[N]:offI[N]], \
				self.V[onI[N]:offI[N]] ]
			except:
				return [None, None, None]
			
	def GetNthOff(self,N):
		#Return the data corresponding to the Nth "off" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		elif N == (self.GetNumberPeriods() - 1):
			offI 	= self.GetEdgesOff()
			return [ self.t[offI[-1]:], self.Rs[offI[-1]:], self.V[offI[-1]:] ]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			try:
				return [ self.t[offI[N]:onI[N+1]], self.Rs[offI[N]:onI[N+1]], \
				self.V[offI[N]:onI[N+1]] ]
			except:
				return [None,None,None]
			
class PhiTrace(PVTrace2):
	'''
	Object to read photovoltage versus time data, convert to surface potential, and fit
	for starting with LED on and ending with it off.
	'''
	def __init__(self, filen, tag, off_txCut, off_tlCut, on_txCut, on_tlCut, T, condModel, phiModel, delim=','):
		PVTrace2.__init__(self,filen, tag, off_txCut, off_tlCut, on_txCut, on_tlCut, delim=delim)
		self.pwr 		= self.V		
		N 				= self.GetNumberPeriods()
		onFit 			= []
		offFit			= []	
		self.condModel 	= condModel 
		self.phiModel 	= phiModel 
		self.T 			= T 
		self.Rs 		= phiModel.ConvertDensity(condModel.ConvertRs(T,self.Rs),T)	 #Rs (Ohm/sq) -> density (cm^-2) -> phi (eV)	
		
		for i in range(N):
			[ti,ri,vi] 	= self.GetNthOn(i)
			if ti is not None and ti.shape[0] > 10 and np.max(ti-ti[0]) > on_tlCut:
				onFit.append(SingleTraceFit(ti,ri,on_txCut,on_tlCut))
				#onFit[-1].PlotFits('b','on %d' % (i))				
			[ti,ri,vi] 	= self.GetNthOff(i)
			if ti is not None and ti.shape[0] > 10 and np.max(ti-ti[0]) > off_tlCut:
				offFit.append(SingleTraceFit(ti,ri,off_txCut,off_tlCut))
				#offFit[-1].PlotFits('k','off %d' % (i))
		self.onFit 		= onFit 
		self.offFit 	= offFit
	
	def GetConductivityModel(self):
		return self.condModel 
		
	def GetPhiModel(self):
		return self.phiModel
		
	def GetTemperature(self):
		return self.T
				
class PolTrace(PVTrace):
	'''
	Object to read and hold polarization dependence data 
	'''

	def __init__(self, filen, tag, delim=','):
		#Angle stored as self.V
		self.tag 	= tag 
		self.filen 	= filen
		[self.t,self.Rs,self.V,self.pwr] 	= self.__LoadFile__(filen, delim)

	def __LoadFile__(self, filen, delim):
		with open(filen, 'rU') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=delim, skipinitialspace=True )
			t 		= []
			Rs	 	= []
			V 		= [] 	
			pwr 	= []
			row 	= spamreader.next()
			try:
				while row and not("Time" in row[0]):
					row 	= spamreader.next()
				for row in spamreader:
					t.append(float(row[0]))					
					Rs.append(float(row[1]))
					V.append(float(row[2]))
					pwr.append(float(row[3]))
			except:
				return [None, None, None, None]
			t 		= np.array(t)
			Rs 		= np.array(Rs)
			V		= np.array(V)
			pwr 	= np.array(pwr)
			return [t, Rs, V, pwr]	
		
	def GetAngle(self):
		return self.V 
		
	def GetPower(self):
		return self.pwr 
		
	def SetFile(self, filen, delim=','):
		self.filen 	= filen
		[self.t,self.Rs,self.V,self.pwr] 	= self.__LoadFile__(filen, delim)
		
	def GetEdgesOn(self):
		#Returns the indeces where the photodiode is turned on
		return np.hstack([0,np.nonzero(np.logical_and(self.pwr[:-1] < 0.1, self.pwr[1:] > 0.1))[0]])
		
	def GetEdgesOff(self):
		#Returns the indeces where the photodiode is turned off
		return np.nonzero(np.logical_and(self.pwr[:-1] > 0.1, self.pwr[1:] < 0.1))[0]
		
	def GetNthOn(self,N):
		#Return the data corresponding to the Nth "on" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			return [ self.t[onI[N]:offI[N]], self.Rs[onI[N]:offI[N]], \
			self.V[onI[N]:offI[N]], self.pwr[onI[N]:offI[N]] ]
			
	def GetNthOff(self,N):
		#Return the data corresponding to the Nth "off" period; [0,N) indexing
		if N >= self.GetNumberPeriods() or N < 0:
			return [None, None, None]
		elif N == (self.GetNumberPeriods() - 1):
			offI 	= self.GetEdgesOff()
			return [ self.t[offI[-1]:], self.Rs[offI[-1]:], self.V[offI[-1]:] ]
		else:
			onI 	= self.GetEdgesOn()
			offI 	= self.GetEdgesOff()
			return [ self.t[offI[N]:onI[N+1]], self.Rs[offI[N]:onI[N+1]], \
			self.V[offI[N]:onI[N+1]], self.pwr[offI[N]:onI[N+1]] ]
			
class SingleTrace:
	'''
	A small class meant to hold a partial cycle of a PV or polarization trace 
	'''				
	def __init__(self, t, Rs):
		self.t0 	= t[0]
		self.t 		= t - t[0]
		self.Rs 	= Rs
		
	def GetT0(self):
		return self.t0 
		
	def GetT(self):
		return self.t 
		
	def GetRs(self):
		return self.Rs 
		
	def SetT0(self,t0):
		self.t 		= self.t + self.t0 - t0	
		self.t0 	= t0 

	def SetT(self,t):
		self.t0 	= t[0] 
		self.t 		= t - t[0]
		
	def SetRs(self,Rs):
		self.Rs 	= Rs 
		
class SingleTraceFit(SingleTrace):

	def __init__(self, t, Rs, tXcutOff, tLcutOn):
		'''
		Class to hold a single trace, fit results, and plot methods
		'''
		SingleTrace.__init__(self,t[1:],Rs[1:])
		self.tXcutOff 	= tXcutOff 
		self.tLcutOn 	= tLcutOn
		self.__FitTraces__(tXcutOff, tLcutOn)
		
	def __FitTraces__(self,tXcutOff,tLcutOn):
		'''
		Fit exponential and logarithmic functions to the on trace in order to extract timescales and Rs(0)
		- Fit the exponential response for 0 < t < tXcutOff
		- Fit the logarithmic response for tLcutOn < t

		Parameters:
		t: time (s)
		Rs: Sheet resistance (Ohm/sq)
		t1cut: Exponential fit for t_1 on 0 <= t <= t1cut (A*exp(-B*t)+C)
		t2cut: Linear regression log_10(t) v. Rs on t2cut <= t < tmax
		
		Returns:
		list of 4 elements:
		ele[0] : linear fit timescale (Ohm/sq-s)
		ele[1] : linear fit Rs(0) (Ohm/sq)
		ele[2] : log fit timescale (Ohm/sq-s)
		ele[3] : log fit timescale error (Ohm/sq-s)		
		'''
		t 		= self.t + 0.1
		self.t 	= t
		rs 		= self.Rs
		def FitFxn(x):
			A 	= x[0]
			B 	= np.abs(x[1])
			C 	= x[2]
			return rs[ii1] - (A*np.exp(-B*t[ii1])+C)
		
		#Fit exponential to retrieve time constant of initial slope
		ii1 		= np.nonzero(t <= tXcutOff)[0]
		[p1,cov1]	= np.polyfit(t[ii1],rs[ii1],1,cov=True)
		p1err 		= np.sqrt(np.diag(cov1))
		A0 			= rs[ii1[0]] - rs[ii1[-1]]
		B0 			= p1[0] / A0
		C0 			= rs[ii1[-1]]
		xi 			= leastsq(FitFxn,[A0,B0,C0])[0]
		self.Ax 	= xi[0]
		self.taux 	= 1 / np.abs(xi[1])
		self.Bx 	= xi[2]
		
		#Fit logarithm to get second time constant
		ii2 		= np.nonzero(t >= tLcutOn)[0]
		def FitFxn(x):
			A 	= x[0]
			B 	= np.abs(x[1])
			C 	= x[2]	
			return rs[ii2] - (A*np.log(B*t[ii2]+C))		
		[p2,cov2] 	= np.polyfit(np.log(t[ii2]), rs[ii2],1,cov=True)
		p2err 		= np.sqrt(np.diag(cov2)) 	
		A0 			= p2[0]
		B0 			= np.exp(p2[1]/p2[0])
		C0 			= 0
		xi 			= leastsq(FitFxn,[A0,B0,C0])[0]
		#self.taulerr = np.sqrt( np.power(p2err[1]*self.taul/self.Al,2) +\
		#np.power(p2err[0]*self.taul*p2[1]/p2[0]**2,2) )
		self.Al 	 = xi[0]
		self.Alerr 	 = 0#p2err[0]
		self.taul 	 = 1/xi[1]	
		self.taulerr = 0
		self.tau0 	 = xi[2] / xi[1]
			
	def GetFitX(self):
		#Return a fit function object for the exponential part of the trace
		return (lambda t: self.Ax * np.exp(-t / self.taux) + self.Bx) 
		
	def GetFitL(self):
		#Return a fit function object for the logarithmic part of the trace	
		#return (lambda t: self.Al * np.log(t / self.taul))
		return (lambda t: self.Al*np.log( (t+self.tau0)/self.taul ))
		
	def PlotFits(self,dcol,label='',xcol='k',lcol='k',MIN=False,NORM=False):
		if MIN:
			div 	= 60.
		else:
			div 	= 1.
		tx 	= np.linspace(0,self.tXcutOff,200) / div
		fx 	= self.GetFitX()
		tl 	= np.logspace(np.log10(self.tLcutOn),np.log10(np.max(self.t)),200) / div
		fl 	= self.GetFitL()
		if NORM:
			Rsmin 	= np.min(self.Rs)
			Rsnorm	= np.max(self.Rs-Rsmin)
			plt.semilogx(self.t/div,(self.Rs-Rsmin)/Rsnorm,'-',color=dcol,label=label)
			plt.semilogx(tx,(fx(tx*div)-Rsmin)/Rsnorm,'--',lw=3,color=xcol)
			plt.semilogx(tl,(fl(tl*div)-Rsmin)/Rsnorm,'--',lw=3,color=lcol)	
			plt.ylim([-0.1,1.1])
		else:		
			plt.semilogx(self.t/div,self.Rs,'-',color=dcol,label=label)
			plt.semilogx(tx,fx(tx*div),'--',lw=3,color=xcol)
			plt.semilogx(tl,fl(tl*div),'--',lw=3,color=lcol)
		
	def GetAx(self):
		return self.Ax
		
	def GetTaux(self):
		return self.taux 
		
	def GetTau0(self):
		return self.tau0
		
	def GetBx(self):
		return self.Bx 
		
	def GetAl(self):
		return self.Al 
		
	def GetAlerr(self):
		return self.Alerr 
		
	def GetTaul(self):
		return self.taul 
		
	def GetTaulerr(self):
		return self.taulerr 
		
	def SetAx(self,Ax):
		self.Ax 	= Ax
		
	def SetTaux(self,taux):
		self.taux 	= taux
		
	def SetBx(self,Bx):
		self.Bx 	= Bx
		
	def SetAl(self,Al):
		self.Al 	= Al
		
	def SetAlerr(self,Alerr):
		self.Alerr 	= Alerr 
		
	def SetTaul(self,taul):
		self.taul 	= taul
		
	def SetTaulerr(self,taulerr):
		self.taulerr = taulerr
			
def MakeOnPlot(tobj,LOGX=False,LOGY=False):
	N 	= tobj.GetNumberPeriods()
	clrs 	= plt.cm.jet(np.linspace(0,1,N))
	for i in range(N):
		ti 				= tobj.GetNthOn(i)[0]
		Rsi 			= tobj.GetNthOn(i)[1]
		if LOGX is True and LOGY is True:
			plt.loglog(ti-ti[0]+1,Rsi,'-',color=clrs[i])		
		elif LOGX is True:
			plt.semilogx(ti-ti[0]+1,Rsi,'-',color=clrs[i])	
		elif LOGY is True:
			plt.semilogy(ti-ti[0],Rsi,'-',color=clrs[i])			
		else:
			plt.plot(ti-ti[0],Rsi,'-',color=clrs[i])
	plt.xlabel('Time (s)')
	plt.ylabel('$R_s$ ($\Omega$/sq)')
	plt.grid()
	
def MakeOffPlot(tobj,LOGX=False,LOGY=False):
	N 	= tobj.GetNumberPeriods()
	clrs 	= plt.cm.jet(np.linspace(0,1,N-1))
	for i in range(1,N):
		ti 				= tobj.GetNthOn(i)[0]
		Rsi 			= tobj.GetNthOn(i)[1]
		if LOGX is True and LOGY is True:
			plt.loglog(ti-ti[0]+1,Rsi,'-',color=clrs[i-1])		
		elif LOGX is True:
			plt.semilogx(ti-ti[0]+1,Rsi,'-',color=clrs[i-1])	
		elif LOGY is True:
			plt.semilogy(ti-ti[0],Rsi,'-',color=clrs[i-1])			
		else:
			plt.plot(ti-ti[0],Rsi,'-',color=clrs[i-1])
	plt.xlabel('Time (s)')
	plt.ylabel('$R_s$ ($\Omega$/sq)')
	plt.grid()	
	
def FitOffTrace(t,rs,t1cut=100.,t2cut=150.,PLOT=False,FITEXP=True):
	'''
	Fit linear regressions to the off trace in order to extract timescales and Rs(0)
	
	Parameters:
	t: time (s)
	Rs: Sheet resistance (Ohm/sq)
	t1cut: Linear regression for t_1 on 0 <= t <= t1cut
	t2cut: Linear regression log_10(t) v. Rs on t2cut <= t < tmax
	
	Returns:
	list of 6 elements:
	ele[0] : linear fit timescale (Ohm/sq-s)
	ele[1] : linear fit timescale std (Ohm/sq-s)	
	ele[2] : linear fit Rs(0) (Ohm/sq)
	ele[3] : linear fit Rs(0) std (Ohm/sq)
	ele[4] : log fit timescale (Ohm/sq-s)
	ele[5] : log fit timescale error (Ohm/sq-s)		
	'''
	t 		= t - t[0] + 0.1
	
	#Fit linear slope to first time constant
	ii1 		= np.nonzero(t <= t1cut)[0]
	def FitFxn(x):
		A 	= x[0]
		B 	= x[1]
		C 	= x[2]
		return rs[ii1] - (A*np.exp(-B*t[ii1])+C)
	
	#Fit exponential to retrieve time constant of initial slope
	ii1 		= np.nonzero(t <= t1cut)[0]
	if FITEXP:
		[p1,cov1]	= np.polyfit(t[ii1],rs[ii1],1,cov=True)
		p1err 		= np.sqrt(np.diag(cov1))
		A0 			= p1[1] - rs[ii1[-1]]
		B0 			= np.abs(p1[0] / A0)
		C0 			= rs[ii1[-1]]
		xi 			= leastsq(FitFxn,[A0,B0,C0])[0]	
		p1 			= np.array([1/np.abs(xi[1]),xi[0]+xi[2]])
		p1err 		= np.array([0,0])
	else:
		[p1,cov1]	= np.polyfit(t[ii1],rs[ii1],1,cov=True)
		p1err 		= np.sqrt(np.diag(cov1))
	
	#Fit logarithmic slope to second time constant
	ii2 		= np.nonzero(t >= t2cut)[0]
	[p2,cov2] 	= np.polyfit(np.log10(t[ii2]), rs[ii2],1,cov=True)
	p2err 		= np.sqrt(np.diag(cov2)) 
	taul 		= np.exp(-p2[1]/p2[0])
	taul_err 	= np.sqrt( np.power(p2err[1]*taul/p2[0],2) +\
	np.power(p2err[0]*taul*p2[1]/p2[0]**2,2) )	
	
	if PLOT:
		#Plot for debugging
		plt.semilogx(t,rs,'k-')
		if FITEXP:
			plt.semilogx(t[ii1],rs[ii1]-FitFxn(xi),'r--',lw=2)		
		else:
			plt.semilogx(t[ii1],np.polyval(p1,t[ii1]),'r--',lw=2)
		plt.semilogx(t[ii2],np.polyval(p2,np.log10(t[ii2])),'b--',lw=2)
		plt.xlabel('Time (s)')
		plt.ylabel('$R_s$ ($\Omega$)')
		plt.grid()
	
		plt.show()
	
	return [p1[0],p1err[0],p1[1],p1err[1],taul,taul_err]
	
def FitAllOffTraces(tobj,PLOT=False):
	'''
	Loop throught and do off trace fits in order to extract timescales and Rs(0) for all 
	periods in tobj
	
	Parameters:
	tobj: <PVTrace> object
	
	Returns:
	list of 6 elements:
	ele[0] : linear fit timescale (Ohm/sq-s)
	ele[1] : linear fit timescale std (Ohm/sq-s)	
	ele[2] : linear fit Rs(0) (Ohm/sq)
	ele[3] : linear fit Rs(0) std (Ohm/sq)
	ele[4] : logarithmic fit timescale log_10(Ohm/sq-s)
	ele[5] : logarithmic fit timescale std log_10(Ohm/sq-s)	
	'''
	
	N 		= tobj.GetNumberPeriods()
	tau1 	= np.zeros(N-1)
	tau1err = np.zeros(N-1)
	Rs0 	= np.zeros(N-1)
	Rs0err 	= np.zeros(N-1)
	tau2 	= np.zeros(N-1)
	tau2err = np.zeros(N-1)
	
	for i in range(1,N):
		[tau1[i-1],tau1err[i-1],Rs0[i-1],Rs0err[i-1],tau2[i-1],tau2err[i-1]] = \
		FitOffTrace(tobj.GetNthOff(i)[0], tobj.GetNthOff(i)[1])
		
	if PLOT:
		plt.figure()
		plt.errorbar(np.arange(1,tau1.shape[0]+1),tau1,yerr=tau1err,fmt='o',color='k')
		plt.xlabel('Period #')
		plt.ylabel('$\\tau_1$ ($\Omega$/sq-s)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])

		plt.figure()
		plt.errorbar(np.arange(1,Rs0.shape[0]+1),Rs0,yerr=Rs0err,fmt='o',color='b')
		plt.xlabel('Period #')
		plt.ylabel('$R_s$ ($\Omega$/sq)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])	
		
		plt.figure()
		plt.errorbar(np.arange(1,tau2.shape[0]+1),tau2,yerr=tau2err,fmt='o',color='r')
		plt.xlabel('Period #')
		plt.ylabel('$\\tau_2$ (log$_{10}\Omega$/sq-s)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])		
		plt.show()
	
	return [tau1,tau1err,Rs0,Rs0err,tau2,tau2err]
	
def FitOnTrace(t,rs,t1cut=30.,t2cut=100,PLOT=False):
	'''
	Fit exponential and logarithmic functions to the on trace in order to extract timescales and Rs(0)
	
	Parameters:
	t: time (s)
	Rs: Sheet resistance (Ohm/sq)
	t1cut: Exponential fit for t_1 on 0 <= t <= t1cut (A*exp(-B*t)+C)
	t2cut: Linear regression log_10(t) v. Rs on t2cut <= t < tmax
	
	Returns:
	list of 4 elements:
	ele[0] : linear fit timescale (Ohm/sq-s)
	ele[1] : linear fit Rs(0) (Ohm/sq)
	ele[2] : log fit timescale (Ohm/sq-s)
	ele[3] : log fit timescale error (Ohm/sq-s)		
	'''
	t 		= t - t[0] + 0.1
	
	def FitFxn(x):
		A 	= x[0]
		B 	= x[1]
		C 	= x[2]
		return rs[ii1] - (A*np.exp(-B*t[ii1])+C)
	
	#Fit exponential to retrieve time constant of initial slope
	ii1 		= np.nonzero(t <= t1cut)[0]
	[p1,cov1]	= np.polyfit(t[ii1],rs[ii1],1,cov=True)
	p1err 		= np.sqrt(np.diag(cov1))
	A0 			= p1[1] - rs[ii1[-1]]
	B0 			= p1[0] / A0
	C0 			= rs[ii1[-1]]
	xi 			= leastsq(FitFxn,[A0,-B0,C0])[0]
	
	#Fit logarithm to get second time constant
	ii2 		= np.nonzero(t >= t2cut)[0]
	[p2,cov2] 	= np.polyfit(np.log10(t[ii2]), rs[ii2],1,cov=True)
	p2err 		= np.sqrt(np.diag(cov2)) 	
	taul 		= np.exp(-p2[1]/p2[0])
	taul_err 	= np.sqrt( np.power(p2err[1]*taul/p2[0],2) +\
	np.power(p2err[0]*taul*p2[1]/p2[0]**2,2) )		
	
	if PLOT:
		#Plot for debugging
		plt.semilogx(t,rs,'k-')
		plt.semilogx(t[ii1],rs[ii1]-FitFxn(xi),'r--',lw=2)
		plt.semilogx(t[ii2],np.polyval(p2,np.log10(t[ii2])),'b--',lw=2)		
		plt.xlabel('Time (s)')
		plt.ylabel('$R_s$ ($\Omega$)')
		plt.grid()
		plt.show()
			
	
	return [np.abs(xi[1]*xi[0]),xi[0]+xi[2],taul,taul_err]
	
def FitAllOnTraces(tobj,PLOT=False):
	'''
	Loop throught and do on trace fits in order to extract timescales and Rs(0) for all 
	periods in tobj
	
	Parameters:
	tobj: <PVTrace> object
	
	Returns:
	list of 4 elements:
	ele[0] : exponential fit timescale (Ohm/sq-s)
	ele[1] : exponential fit Rs(0) (Ohm/sq)
	ele[2] : logarithmic fit timescale log_10(Ohm/sq-s)
	ele[3] : logarithmic fit timescale std log_10(Ohm/sq-s)	
	'''	
	N 		= tobj.GetNumberPeriods()
	tau1 	= np.zeros(N-1)
	tau1err = np.zeros(N-1)
	Rs0 	= np.zeros(N-1)
	Rs0err 	= np.zeros(N-1)
	tau2 	= np.zeros(N-1)
	tau2err = np.zeros(N-1)
	
	for i in range(1,N):
		[tau1[i-1],Rs0[i-1],tau2[i-1],tau2err[i-1]] = \
		FitOnTrace(tobj.GetNthOn(i)[0], tobj.GetNthOn(i)[1])
		
	if PLOT:
		plt.figure()
		plt.plot(np.arange(1,tau1.shape[0]+1),tau1,'o',color='k')
		plt.xlabel('Period #')
		plt.ylabel('$\\tau_1$ ($\Omega$/sq-s)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])

		plt.figure()
		plt.plot(np.arange(1,Rs0.shape[0]+1),Rs0,'o',color='b')
		plt.xlabel('Period #')
		plt.ylabel('$R_s$ ($\Omega$/sq)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])	
		
		plt.figure()
		plt.errorbar(np.arange(1,tau2.shape[0]+1),tau2,yerr=tau2err,fmt='o',color='r')
		plt.xlabel('Period #')
		plt.ylabel('$\\tau_2$ (log$_{10}\Omega$/sq-s)')
		plt.grid()
		plt.xlim([0,tau1.shape[0]+1])		
		plt.show()		
		
		plt.show()
	
	return [tau1,Rs0,tau2,tau2err]
	
class ConductivityModel:
	'''
	Object that converts between sheet resistance and 2DEG density 
	'''
	def __init__(self,T,mu):
		self.T 		= T 
		self.mu 	= mu
		self.model 	= interp1d(T,mu,bounds_error=False,fill_value="extrapolate")
		
	def GetT(self):
		return T 
		
	def GetMu(self):
		return mu 
		
	def PlotModel(self):
		plt.plot(self.T,self.mu,'ko-',ms=8,lw=1.5)
		plt.xlabel('Temperature (K)')
		plt.ylabel('Mobility (cm$^2$/V-s)')
		
	def ConvertRs(self,T,Rs):
		return 1. / (1.609e-19 * Rs * self.model(T))
		

class ConductivityModelMeasure1(ConductivityModel):

	def __init__(self):
		#Temperature-dependent mobility
		T 			= np.array([295,302,320,341,360,380,400]) #K
		mu 			= np.array([962,932,857,793,718,640,568]) #cm^2/V-s
		ConductivityModel.__init__(self,T,mu)
		
class ChargeModel:
	'''
	Object that converts between 2DEG density and surface potential
	'''
	
	def __init__(self,dGaN,dInAlN,dAlN):
		#thicknesses are in nm to be converted to m 
		self.dGaN 				= dGaN * 1e-9
		self.dAlN 				= dAlN * 1e-9
		self.dInAlN 			= dInAlN * 1e-9 
		[self.phi,self.n2deg] 	= self.__CalculateModel__() #(eV, cm^{-2})
		model 					= interp1d(self.n2deg,self.phi,bounds_error=False,fill_value="extrapolate")
		def modfxn(n,T):
			return model(n)
		self.model 				= modfxn
		
	def PlotModel(self,T=295,clr='k'):
		plt.plot(self.n2deg,self.model(self.n2deg,T),'-',color=clr,lw=1.5)
		plt.xlabel('$n_{2D}$ (cm$^{-2}$)')
		plt.ylabel('$\phi_b$ (eV)')
		#plt.grid()
		#plt.show()
		
	def GetPhi(self):
		return self.phi 
		
	def GetN(self):
		return self.n2deg 
		
	def ConvertDensity(self,n,T):
		#Evaluate the model for the np.ndarray of densities <n>
		return self.model(n)
		
	def WriteToCSV(self,filen):
		with open(filen, 'wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow(['n2deg(cm^-2)','phi_b(eV)'])
			for i in range(self.n2deg.shape[0]):
				spamwriter.writerow(['%0.3e' % self.n2deg[i],'%0.4f' % (self.phi[i])])
		
	def __CalculateModel__(self,nmin=2e12,nmax=5e13):
		dGaN 		= self.dGaN
		dAlN 		= self.dAlN 
		dInAlN 		= self.dInAlN 
		n2deg		= np.logspace(np.log10(nmin),np.log10(nmax),1000) * 1e4 #charge densities to calculate for
		eV2J 		= 1.602e-19 #energy conversion 

		#All units SI, to be converted at end 
		kB 					= 1.38e-23  #Boltzmann constant 
		T 					= 295 		#Ambient temperature
		e0 					= 1.602e-19 #Coulomb
		hbar 				= 1.055e-34 #Planck constant 
		m0 					= 9.109e-31 #electron rest mass 
		mGaN 				= 0.20 * m0 #electron mass in GaN 
		eps0 				= 8.85e-12 #static relative permittivity
		epsGaN 				= 9.5 
		epsAlN 				= 8.57
		epsInAlN 			= 10.1
		dEc_GaN_InAlN 		= 0.9 * eV2J #conduction band offsets
		dEc_InAlN_AlN 		= 0.8 * eV2J 
		dEc_GaN_AlN 		= 1.7 * eV2J
		sig_GaN_AlN 		=  6.64e13 * 1e4 #spontaneous polarization charge
		sig_InAlN_AlN 		= -3.57e13 * 1e4 
		sig_GaN_InAlN 		=  -3.07e13 * 1e4
		
		#Second energy level -- small correction
		n2deg1 				= kB * T * mGaN * np.log( np.exp( (0.01)*eV2J / (kB*T) ) + 1) / (pi*hbar**2)
		n2deg0 				= n2deg - n2deg1
		
		#Calculate the terms individually
		E0 					= pi*hbar**2*n2deg/mGaN 
		Ef_E0 				= np.power(9*pi*hbar*e0**2*n2deg/(8.*eps0*epsGaN*sqrt(8*mGaN)),2./3.)
		F_AlN 				= e0*(sig_GaN_AlN - n2deg) / (eps0*epsAlN)
		F_InAlN				= e0*(sig_InAlN_AlN + sig_GaN_AlN - n2deg) / (eps0*epsInAlN)
		F_GaN 				= e0*(sig_GaN_InAlN + sig_InAlN_AlN + sig_GaN_AlN - n2deg) / (eps0*epsGaN)
		
		#Do the sum
		phib 				= -(-e0*F_GaN*dGaN + dEc_GaN_InAlN - e0*F_InAlN*dInAlN + dEc_InAlN_AlN \
		- e0*F_AlN*dAlN - dEc_GaN_AlN + E0 + Ef_E0)/eV2J
		
		n2deg 				= n2deg[ np.logical_and(phib > 0, phib < 3.46) ]
		phib 				= phib[ np.logical_and(phib > 0, phib < 3.46) ]
		
		return [phib, n2deg / 1e4]
		
class EmpiricalChargeModel(ChargeModel):

	def __init__(self,dGaN,dInAlN,dAlN,T1,n0_1,T2,n0_2):
		'''
		This model takes the NextNano / charge model slope and uses the experimentally determined
		x-intercept (n_2DEG such that phi_bb=0) as a function of temperature to calibrate the conversion
		'''
		ChargeModel.__init__(self,dGaN,dInAlN,dAlN)
		def SimpleModel(n,T):
			n0_0 	= n0_1 + (T-T1) * (n0_2 - n0_1) / (T2 - T1)
			return (n0_0 - n) / 7.2e12
		self.model 	= SimpleModel

	def ConvertDensity(self,n,T):
		return self.model(n,T)
		
def PlotNLevel(T,clr):
	eV2J 		= 1.609e-19
	m0 			= 9.11e-31
	hbar 		= 1.055e-34 #Planck constant 	
	EimEf 		= np.linspace(-0.2,0.4,100)
	kB 			= 1.38e-23  #Boltzmann constant 
	mGaN 		= 0.20 * m0 #electron mass in GaN 	
	n2d 		= 1e-4*kB * T * mGaN * np.log( np.exp( eV2J*EimEf / (kB*T) ) + 1) / (pi*hbar**2)
	plt.semilogy(EimEf,n2d,'-',lw=1.5,color=clr,label='%d K' % (T))
	return [EimEf,n2d]
			
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 30,
	'figure.autolayout':True, 'figure.figsize':[1.2*8,8]}	#*2.0/(math.sqrt(5)+1)
	mpb.rcParams.update(params)	
	
	q 		= 1.609e-19
	sl 		= -0.72e13  #cm^-2-eV^-1
	mu 		= 925 #cm^2/V-s
	Rs 		= 420 
	s_Rs	= 5. #Error in sheet resistance
	s_mu 	= 0.1*mu  #error in mobility 
	s_rho 	= 0.1*1.05e13 #error in carrier density 
	
	sig_phi 	= -np.sqrt( (1/(q*mu**2*Rs))**2 + (1/(q*mu*Rs**2))**2 + s_rho**2 ) / sl
	print sig_phi
	
	[E,n295] 	= PlotNLevel(295,'k')
	#[_,n365] 	= PlotNLevel(365,'b')	
	[_,n400] 	= PlotNLevel(400,'r')
	plt.semilogy(E,n400-n295,'g-',lw=1.5,label='Difference')
	#plt.semilogy(E,n365-n295,'m-',lw=1.5)	
	plt.xlabel('$E_i - E_f$ (eV)')
	plt.ylabel('$n_{2D}$ (cm$^{-2}$)')
	plt.grid()
	plt.xlim([-0.2,0.4])
	plt.legend(loc='lower left',prop={'size':24})
	plt.savefig('20171205_charge_density.png',bbox_inches='tight')
	plt.savefig('20171205_charge_density.pdf',bbox_inches='tight')	
	plt.show()

	mymodel 	= ChargeModel(2.,3.,1.)
	#mymodel.PlotModel()
	eneV 		= np.linspace(0,1.4,1000)
	plt.plot((-0.72*eneV+1.60)*1e13,eneV,'k-',lw=2)
	plt.plot((-0.72*0.68+1.60)*1e13,0.68,'bs',lw=2,ms=16)
	plt.plot((-0.72*0.78+1.60)*1e13,0.78,'ro',lw=2,ms=16)	
	#mymodel.WriteToCSV('charge_model.csv')
	plt.grid()
	plt.xlabel('$n_{2D}$ (cm$^{-2}$)')
	plt.ylabel('$\phi_b$ (eV)')
	#plt.legend(['Simple charge','NextNano'],loc='upper right',prop={'size':24})
	plt.savefig('charge_model_fixed.png',bbox_inches='tight')
	plt.savefig('charge_model_fixed.pdf',bbox_inches='tight')	
	plt.show()
	
	#filen 	= "U:/2DEG/Data/20171002-Day6/Blue460nm_LED_50uA_RH_15per_23C_40minOn_40minOff_LED_3p3V.csv"
	#Blue460nm_LED_50uA_RH_15per_23C_5minOn_10minOff_LED_2p8V.csv"
	#t1 	= PVTrace(filen, "460 nm 3.2 V")
	
	#FitOnTrace(t1.GetNthOn(1)[0],t1.GetNthOn(1)[1])
	#FitAllOnTraces(t1)