from uPL import *
import matplotlib as mpb
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def FitTimeSeries(ps,fname=None,LAMTOL=0.5,tmin=-1,tmax=-1):
	#Sort peaks in descending order of power
	[pwr,pks] 	= __FitWaterfall__(ps,tmin,tmax)
	ii 		= np.argsort(pwr)
	pwr 	= pwr[ii]
	pks 	= map(lambda x: pks[x], ii)
	#Do fits and extract values
	lam0 	= map(lambda x: np.array(map(lambda y: y.GetLambda0(), x)), pks)
	fwhm 	= map(lambda x: np.array(map(lambda y: y.GetFWHM(), x)), pks)
	a 		= map(lambda x: np.array(map(lambda y: y.GetA(), x)), pks)
	N 		= np.max(np.array(map(lambda x: x.shape[0], lam0)))
	lamArr 	= np.zeros((pwr.shape[0],N))
	fwhmArr	= np.zeros((pwr.shape[0],N))
	aArr 	= np.zeros((pwr.shape[0],N))
	#Organize into different modes
	for i in range(len(pks)):
		lami 	= lam0[i]
		if lami.shape[0] > 1:
			ii 		= np.argsort(lami)
			lami 	= np.sort(lami)
			fwhmi	= fwhm[i][ii]
			ai 		= a[i][ii]
		else:
			fwhmi 	= fwhm[i]
			ai 		= a[i]
		if i == 0:
			lamArr[i,:lami.shape[0]] 	= lami
			fwhmArr[i,:lami.shape[0]] 	= fwhmi 
			aArr[i,:lami.shape[0]] 	= ai 
		else:
			lam00 	= lamArr[i-1,:]
			for j in range(lami.shape[0]):
				lamj 	= lami[j]
				jj 		= np.argmin(np.abs(lam00-lamj))
				if np.abs(lamj-lam00[jj]) < LAMTOL:
					lamArr[i,jj]	= lamj
					fwhmArr[i,jj] 	= fwhmi[j]
					aArr[i,jj] 		= ai[j]
				else:
					jj 	= np.nonzero(lam00==0)[0]
					if not jj.shape[0] == 0:
						lamArr[i,jj]	= lamj
						fwhmArr[i,jj] 	= fwhmi[j]
						aArr[i,jj] 		= ai[j]
	return [pwr, lamArr, fwhmArr, aArr]
						
def PlotFitTimeSeries(ps,fname=None,LAMTOL=0.5,tmin=-1,tmax=-1):
	[pwr, lamArr, fwhmArr, aArr] 	= FitTimeSeries(ps,fname,LAMTOL,tmin,tmax)
	#Make a fancy plot
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 28,
	'figure.autolayout':True, 'figure.figsize':[12,12*2.0/(math.sqrt(5)+1)]}		
	mpb.rcParams.update(params)		
	plt.figure()
	clrs 				= plt.cm.jet(np.linspace(0, 1, lamArr.shape[1]))
	pmax 				= np.min(pwr)
	plt.plot(ps.GetWavelength(),ps.GetSingleSpectrum(pmax),'k-',lw=1.5,label='P = %d %s' % (pmax,ps.GetUnits()))
	nmn 	= np.min(ps.GetSingleSpectrum(pmax))
	nmx 	= np.max(ps.GetSingleSpectrum(pmax))
	for i in range(lamArr.shape[1]):
		plt.plot(lamArr[0,i]*np.ones(100),np.logspace(np.log10(nmn),np.log10(nmx),100),'--',lw=1.5,color=clrs[i],label="Mode %d" % (i+1))
	plt.legend(loc='lower left',prop={'size':16})
	plt.grid()
	plt.xlim([np.min(ps.GetWavelength()),np.max(ps.GetWavelength())])
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts')
	if fname is not None:
		plt.savefig('%s_spect.png' % (fname),bbox_inches='tight')
		plt.savefig('%s_spect.pdf' % (fname),bbox_inches='tight')
		plt.clf()
	
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 18,
	'figure.autolayout':True, 'figure.figsize':[12*2.0/(math.sqrt(5)+1),16]}		
	mpb.rcParams.update(params)	
	plt.figure()
	ax1 				= plt.subplot(311)
	for i in range(lamArr.shape[1]):
		if np.any(np.nonzero(lamArr[:,i])):
			ii 		= np.nonzero(lamArr[:,i])
			maxA 	= np.max(aArr[ii,i])
			plt.plot( pwr[ii]/60., aArr[ii,i][0,:]/maxA, 'o--', lw=1.5, ms=4, color=clrs[i] )
	plt.ylabel('Normalized intensity')
	plt.setp(ax1.get_xticklabels(), visible=False)	
	plt.grid()
	
	ax1 				= plt.subplot(312)
	for i in range(lamArr.shape[1]):
		if np.any(np.nonzero(lamArr[:,i])):
			ii 		= np.nonzero(lamArr[:,i])
			pwri 	= pwr[ii]
			lami 	= lamArr[ii,i][0,:]
			lamMn 	= lami[0]
			plt.plot( pwri/60., lami - lamMn , 'o--', lw=1.5, ms=4, color=clrs[i] )
	plt.ylabel('$\Delta \lambda$ (nm)')
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.grid()	
	
	ax1 				= plt.subplot(313)
	for i in range(lamArr.shape[1]):
		if np.any(np.nonzero(lamArr[:,i])):
			ii 		= np.nonzero(lamArr[:,i])
			pwri 	= pwr[ii]
			fwhmi 	= fwhmArr[ii,i][0,:]
			fwhmMn 	= fwhmi[np.argmin(pwri)]
			plt.plot( pwri/60., fwhmi - fwhmMn , 'o--', lw=1.5, ms=4, color=clrs[i] )
	plt.ylabel('$\Delta$FWHM (nm)')
	plt.xlabel('Time (min)')
	plt.grid()
	if fname is not None:
		plt.savefig('%s_stats.png' % (fname),bbox_inches='tight')
		plt.savefig('%s_stats.pdf' % (fname),bbox_inches='tight')
		plt.clf()	
		
def Plot2DTimeseries(ps,NORM=False,ENERGY=True,CMAP=None,CBAR=True,toff=0,tmin=-1,tmax=-1,FLATTEN=False,Nlevels=31,\
	Emin=None,Emax=None,SPIKEFILTER=None,XSCALE=60.,LOG=False):
	t 		= ps.GetPowers()
	if not (tmin == -1):
		t  	= t[t >= tmin]
	if not (tmax == -1):
		t  	= t[t <= tmax]	
	wl 		= ps.GetWavelength()
	if CMAP is None:
		clr 	= plt.cm.jet 
	else:
		clr 	= CMAP
	if ENERGY:
		x 	= 1239.84 / wl 
	else:
		x 	= wl
	dat 	= np.zeros((t.shape[0],wl.shape[0]))
	p 		= None
	for i in range(t.shape[0]):
		dati 		= (ps.GetSingleSpectrum(t[i])-990.)
		if SPIKEFILTER is not None:
			dati 		= signal.medfilt(dati,SPIKEFILTER)
		if FLATTEN:
			if Emin is not None and Emax is not None:
				ii 		= np.nonzero(np.logical_and(x >= Emin, x <= Emax))[0]
				p 		= np.polyfit(x[ii],dati[ii],3)
				dati 	= dati - np.polyval(p,x)
				dati 	= dati - np.min(dati[ii])
			else:
				p 		= np.polyfit(x,dati,2)
				dati 	= dati - np.polyval(p,x)
		if NORM:
			#plt.plot(x,GetBackground1(x,dati,ROLL=-200))
			#plt.plot(x,dati,'r-')
			#plt.show()
			if FLATTEN and Emin is not None and Emax is not None:
				dati 	= dati - np.min(np.min(dati))
				dati 	= dati / np.max(dati[ii])
			else:
				dati 	= dati/np.max(dati)
		dat[i,:] 	= dati
		#dat 		= dat
	#v = np.round(np.linspace(0.6, 1.1, 10, endpoint=True),2)
	dat 	= dat - np.min(dat)
	#dat 	= dat / np.max(np.max(dat))
	if NORM:
		levels 	= np.linspace(0,1,Nlevels)
	else:
		levels 	= np.linspace(np.min(dat),np.max(dat),Nlevels)
	if LOG:
		dat 	= dat / np.max(np.max(dat))
		levels 	= np.linspace(-3,0,Nlevels)
		cs	= plt.contourf((t+toff)/XSCALE,x,np.log10(dat.transpose()+1e-4),levels,cmap=clr)#,vmin=0,vmax=1.1)	
	else:
		cs	= plt.contourf((t+toff)/XSCALE,x,dat.transpose(),levels,cmap=clr)#,vmin=0,vmax=1.1)
	if ENERGY:
		plt.ylabel('Energy (eV)')
	else:
		plt.ylabel('Wavelength (nm)')
	if CBAR:
		cbar = plt.colorbar(cs)
		if LOG:
			cbar.ax.set_ylabel('log(PL Intensity) (arb. units)')		
		else:
			cbar.ax.set_ylabel('PL Intensity (arb. units)')
		if NORM:
			cbar.set_ticks(np.linspace(0,1,11))
	plt.xlabel('Time (min.)')
	if Emin is not None and Emax is not None:
		plt.ylim([Emin,Emax])
	return dat.transpose()
		
def __FitWaterfall__(ps,tmin=-1,tmax=-1):
	pwr 	= ps.GetTimes()
	if not (tmin == -1):
		pwr  	= pwr[pwr >= tmin]
	if not (tmax == -1):
		pwr  	= pwr[pwr <= tmax]
	vpwrs 	= []
	dirout 	= "%s_Fits" % (ps.GetTag())
	pkvec 	= []
	if not os.path.exists(dirout):
		os.makedirs(dirout)	
	for pj in pwr:
		figname 	= "%s/%s_%d%s.png" % (dirout,ps.GetTag(),pj,ps.GetUnits())
		up1 		= ps.GetSingleUPL(pj)
		if up1 is not None:
			print "\nFitting: %0.1f %s" % (pj,ps.GetUnits())
			peaks 		= IdentifyPeaks(up1,SAVEFIG=figname)
			if peaks is not None:
				pkvec.append(peaks)
				vpwrs.append(pj)
	return [np.array(vpwrs),pkvec]

def PlotWaterfall(ps,PLOTEVERY=1,tmin=-1,tmax=-1,LOG=False,BIN=0):
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(math.sqrt(5)+1)]}	
	mpb.rcParams.update(params)	
	plt.figure()
	pwr 	= ps.GetPowers()
	if not (tmin == -1):
		pwr  	= pwr[pwr >= tmin]
	if not (tmax == -1):
		pwr  	= pwr[pwr <= tmax]	
	i 		= 0
	units 	= ps.GetUnits()
	if not (PLOTEVERY == 1):
		pwr 	= pwr[::PLOTEVERY]
	clrs 	= plt.cm.jet(np.linspace(0, 1, pwr.shape[0]))	
	scale = 0.5
	for pj in pwr:
		if BIN > 0:
			cts 	= np.convolve(ps.GetSingleSpectrum(pj)-950,np.ones(BIN),'same')/BIN
		else:
			cts 	= (ps.GetSingleSpectrum(pj)-950)
		if LOG:
			cts 	= cts * scale 
			scale 	= 2*scale
		else:
			cts 	= cts + scale 
			scale 	= scale + 0.5*(np.max(cts)-np.min(cts))
		if LOG:
			plt.semilogy(ps.GetWavelength(),cts,'-', lw=1.5, color=clrs[i], label="%0.1f %s" % (pj,units))
		else:
			plt.plot(ps.GetWavelength(),cts,'-', lw=1.5, color=clrs[i], label="%0.1f %s" % (pj,units))		
		i 	= i + 1 
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts (arb)')
	plt.grid()

if __name__=="__main__":
	filen 	= "U:/Spectroscopy/QOLab/20170309-A3530-1/L431-77K_2_2_40uW_Pseries_debugged_down.csv"
	pseries = TimeSeries(filen,0.05,1800,"A3530-77K-L431-2-2-Up")
	PlotFitTimeSeries(pseries,"M2_1_1_40uW_RT")