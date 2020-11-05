from uPL import *
import matplotlib as mpb
import matplotlib.pyplot as plt
import numpy as np
import dill

def PlotFitWaterfall(ps,NAVG=1,fname=None,LAMTOL=0.5,SPIKEFILTER=None,ENERGY=True,FP=False):
	#Sort peaks in descending order of power
	[pwr,pks] 	= FitWaterfall(ps,NAVG=NAVG,FP=FP,SPIKEFILTER=SPIKEFILTER)
	ii 		= np.argsort(pwr)[-1::-1]
	pwr 	= pwr[ii]
	pks 	= map(lambda x: pks[x], ii)
	#Do fits and extract values
	lam0 	= map(lambda x: np.array(map(lambda y: y.GetLambda0(), x)), pks)
	fwhm 	= map(lambda x: np.array(map(lambda y: y.GetFWHM(), x)), pks)
	a 		= map(lambda x: np.array(map(lambda y: y.GetA(), x)), pks)
	bg 		= []
	for blah in pks:
		tmp 	= []	
		for pki in blah:
			fxni 	= pki.GetBackgroundFunction()
			lami 	= pki.GetLambda0()
			tmp.append(fxni(lami))
		bg.append(np.array(tmp))
	N 		= np.max(np.array(map(lambda x: x.shape[0], lam0)))
	lamArr 	= np.zeros((pwr.shape[0],N))
	fwhmArr	= np.zeros((pwr.shape[0],N))
	aArr 	= np.zeros((pwr.shape[0],N))
	bgArr 	= np.zeros((pwr.shape[0],N))
	#Organize into different modes
	for i in range(len(pks)):
		lami 	= lam0[i]
		if lami.shape[0] > 1:
			ii 		= np.argsort(lami)
			lami 	= np.sort(lami)
			fwhmi	= fwhm[i][ii]
			ai 		= a[i][ii]
			bgi 	= bg[i][ii]
		else:
			fwhmi 	= fwhm[i]
			ai 		= a[i]
			bgi 	= bg[i]
		if i == 0:
			lamArr[i,:lami.shape[0]] 	= lami
			fwhmArr[i,:lami.shape[0]] 	= fwhmi 
			aArr[i,:lami.shape[0]] 		= ai
			bgArr[i,:lami.shape[0]] 	= bgi
		else:
			lam00 	= lamArr[i-1,:]
			for j in range(lami.shape[0]):
				lamj 	= lami[j]
				jj 		= np.argmin(np.abs(lam00-lamj))
				if np.abs(lamj-lam00[jj]) < LAMTOL:
					lamArr[i,jj]	= lamj
					fwhmArr[i,jj] 	= fwhmi[j]
					aArr[i,jj] 		= ai[j]
					bgArr[i,jj] 	= bgi[j]
				else:
					jj 	= np.nonzero(lam00==0)[0]
					if not jj.shape[0] == 0:
						lamArr[i,jj]	= lamj
						fwhmArr[i,jj] 	= fwhmi[j]
						aArr[i,jj] 		= ai[j]
						bgArr[i,jj] 	= bgi[j]
	#Make a fancy plot
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 28,
	'figure.autolayout':True, 'figure.figsize':[12,12*2.0/(math.sqrt(5)+1)]}		
	mpb.rcParams.update(params)		
	plt.figure()
	clrs 				= plt.cm.jet(np.linspace(0, 1, lamArr.shape[1]))
	pmax 				= np.max(pwr)
	plt.plot(ps.GetWavelength(),ps.GetSingleSpectrum(pmax),'k-',lw=1.5,label='P = %d %s' % (pmax,ps.GetUnits()))
	nmn 	= np.min(ps.GetSingleSpectrum(pmax))
	nmx 	= np.max(ps.GetSingleSpectrum(pmax))
	for i in range(lamArr.shape[1]):
		plt.plot(lamArr[0,i]*np.ones(100),np.logspace(np.log10(nmn),np.log10(nmx),100),'--',lw=1.5,color=clrs[i],label="Mode %d" % (i+1))
	plt.legend(loc='upper left',prop={'size':16})
	plt.grid()
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Counts')
	plt.xlim([np.min(ps.GetWavelength()),np.max(ps.GetWavelength())])
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
			if not ps.GetScaling():
				scale 	= np.ones(pwr.shape)
			else:
				scale 	= pwr / np.min(pwr)
			ii 		= np.nonzero(lamArr[:,i])
			maxA 	= np.max(aArr[ii,i])
			plt.loglog( pwr[ii], scale[ii]*aArr[ii,i][0,:]/maxA, 'o--', lw=1.5, ms=4, color=clrs[i] )
	plt.ylabel('Normalized intensity')
	plt.setp(ax1.get_xticklabels(), visible=False)	
	plt.grid()
	
	ax1 				= plt.subplot(312)
	for i in range(lamArr.shape[1]):
		if np.any(np.nonzero(lamArr[:,i])):
			ii 		= np.nonzero(lamArr[:,i])
			pwri 	= pwr[ii]
			lami 	= lamArr[ii,i][0,:]
			lamMn 	= lami[np.argmin(pwri)]
			ei 		= 1240./lami 
			eMn 	= 1240./lamMn
			if ENERGY:
				plt.semilogx( pwri, 1e3*(ei - eMn) , 'o--', lw=1.5, ms=4, color=clrs[i] )
			else:
				plt.semilogx( pwri, lami - lamMn , 'o--', lw=1.5, ms=4, color=clrs[i] )
	if ENERGY:
		plt.ylim([-2,2])	
		plt.ylabel('$\Delta E$ (meV)')	
	else:
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
			lami 	= lamArr[ii,i][0,:]		
			lamMn 	= lami[np.argmin(pwri)]			
			ei 		= 1e6*1240.*fwhmi/np.power(lami,2)
			emn 	= 1e6*1240.*fwhmMn/np.power(lamMn,2)
			if ENERGY:
				plt.semilogx( pwri, ei , 'o--', lw=1.5, ms=4, color=clrs[i] )			
			else:
				plt.semilogx( pwri, 1000*fwhmi  , 'o--', lw=1.5, ms=4, color=clrs[i] )
	if ENERGY:	
		plt.ylabel('FWHM ($\mu$eV)')
	else:
		plt.ylabel('FWHM (pm)')
	plt.xlabel('Power (%s)' % (ps.GetUnits()))
	plt.grid()
	if fname is not None:
		plt.savefig('%s_stats.png' % (fname),bbox_inches='tight')
		plt.savefig('%s_stats.pdf' % (fname),bbox_inches='tight')
		plt.clf()	

	dill.dump({'p':pwr,'a':aArr,'lam':lamArr,'fwhm':fwhmArr,'bg':bgArr},open("%s_data.pkl" % (fname),'wb'))
	
		
def FitWaterfall(ps,NAVG=1,SPIKEFILTER=None,FP=False):
	pwr 	= ps.GetPowers()
	vpwrs 	= []
	dirout 	= "%s_Fits" % (ps.GetTag())
	pkvec 	= []
	if not os.path.exists(dirout):
		os.makedirs(dirout)	
	
	for ii in np.arange(0,pwr.shape[0],NAVG):
		if (ii+NAVG) <= pwr.shape[0]:
			jj 			= np.arange(ii,ii+NAVG)
		else:
			jj 			= np.arange(ii,pwr.shape[0])
		up1 			= ps.GetSingleUPL(pwr[jj[0]])
		cts 			= up1.GetCounts()
		ctr 			= 1.0 
		for j in jj[1:]:
			up1 		= ps.GetSingleUPL(pwr[j])
			cts 		= up1.GetCounts() + cts
			ctr 		= ctr + 1.0
		cts				= cts / ctr
		up1.SetCounts(cts)
		pj 			= pwr[jj[0]]
		figname 	= "%s/%s_%d%s.png" % (dirout,ps.GetTag(),pj,ps.GetUnits())
		if up1 is not None:
			print "\nFitting: %0.1f %s" % (pj,ps.GetUnits())
			peaks 		= IdentifyPeaks(up1,SAVEFIG=figname,SIGCUT=2.,SPIKEFILTER=SPIKEFILTER,exclWL=np.linspace(435.78,435.86,20),FP=FP)
			if peaks is not None:
				pkvec.append(peaks)
				vpwrs.append(pj)
	return [np.array(vpwrs),pkvec]

def PlotWaterfall(ps,ENERGY=False,LOG=False,SPIKEFILTER=None,NORMALIZE=False,WLMIN=None,WLMAX=None,SHIFT=0,XSHIFT=0.,WEIGHTS=None,DK=0.):
	#params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	#'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(math.sqrt(5)+1)]}	
	#mpb.rcParams.update(params)	
	plt.figure()
	pwr 	= np.sort(ps.GetPowers())
	clrs 	= plt.cm.jet(np.linspace(0, 1, pwr.shape[0]))
	i 		= 0
	units 	= ps.GetUnits()
	if ENERGY:
		xax 	= 1240./ps.GetWavelength()
		plt.xlabel('Energy (eV)')
	else:
		xax 	= ps.GetWavelength()
		plt.xlabel('Wavelength (nm)')
	for pj in pwr:
		if ps.GetScaling():	
			scale 	= ps.GetIntegrationTime() * pj / np.min(pwr)
		else:
			scale 	= 1
		if WEIGHTS is not None:
			scale = WEIGHTS[i]			
		cts 	= ps.GetSingleSpectrum(pj) - DK		
		if SPIKEFILTER is not None:
			norm_pass 	= 0.3
			#Get the normalized cutoff frequency
			norm_stop 	= 0.5
			#Apply a low-pass filter to get rid of spurious peaks
			#(N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
			#(b, a) 	= signal.butter(N, Wn, btype='low', analog=0, output='ba')	
			cts 	= signal.medfilt(cts,SPIKEFILTER)
			#cts 	= signal.lfilter(b, a, cts)	
			#cts 	= cts[10:-10]
			xplt 	= xax#[10:-10]
		else:
			xplt 	= xax
		if NORMALIZE:
			cts 	= cts - np.min(cts)
			cts 	= cts / np.max(cts)
		else:
			cts 	= cts*scale
		cts 		= cts + i*SHIFT*(np.max(cts)-np.min(cts))
		if LOG:
			plt.semilogy(xplt+XSHIFT,cts,'-', lw=1.5, color=clrs[i], label="%0.1f %s" % (pj,units))
		else:
			plt.plot(xplt+XSHIFT,cts,'-', lw=1.5, color=clrs[i], label="%0.1f %s" % (pj,units))
		i 	= i + 1 

	plt.ylabel('PL Intensity (arb. units)')
	plt.grid()
	plt.xlim([np.min(xax),np.max(xax)])

if __name__=="__main__":
	filen 	= "U:/Spectroscopy/QOLab/20170309-A3530-1/L431-77K_2_2_40uW_Pseries_debugged_down.csv"
	pseries = PowerSweep(filen,0.05,1800,"A3530-77K-L431-2-2-Up")
	PlotFitWaterfall(pseries,"L431-2-2-Down")

	