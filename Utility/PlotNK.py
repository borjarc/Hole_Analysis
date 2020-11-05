import numpy as np
import math
import os.path 
import sys
import csv
import matplotlib as mpb
import matplotlib.pyplot as plt
import dill

def PlotMaterial(label,c1,c2,c3,emin,emax,clr):
	wlv 	= 0.001*np.linspace(1240./emax,1240./emin,150)
	n 		= c1 + c2 / np.power(wlv,2) + c3 / np.power(wlv,4)
	plt.plot(1240./(1000*wlv), n, '-', lw=2, label=label, color=clr)

def PlotSellmeier(label,a,b,c,emin,emax,clr):
	wlv 	= np.linspace(1240./emax,1240./emin,150)
	n 		= np.sqrt(a + b*np.power(wlv,2) / (np.power(wlv,2) - c**2) )
	plt.plot(1240./wlv, n, '--', lw=2, color=clr)
	
def tick_function(locs):
	wl 	= 1240. / locs 
	return map(lambda x: "%d" % (x), wl)
	
def ComputeGVD(lam0,A,B,C):
	c0 		= 3.0e8 #(m/s)
	om0		= 2*math.pi*c0/(lam0*1e-9)
	B 		= B*1e-12 
	C 		= C*1e-24
	dndom 	= 2*B*om0/(2*math.pi*c0)**2 + 4*C*np.power(om0,3)/(2*math.pi*c0)**4
	dn2dom2 = 2.*B/(2*math.pi*c0)**2 + 12.*C*np.power(om0,2)/(2*math.pi*c0)**4
	return (2*dndom/c0 + om0*dn2dom2/c0)
	
if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 30,
	'figure.autolayout':True, 'figure.figsize':[12,12*2.0/(math.sqrt(5)+1)]}	
	mpb.rcParams.update(params)
	
	fig 		= plt.figure()
	ax1 		= plt.gca()
	
	lam 	= np.linspace(1500,360,1000)
	gvd 	= ComputeGVD(lam,2.266,2.13e-2,3.85e-3)
	print gvd[np.argmin(np.abs(lam-405.))]
	print 2.42 * 100e-6 / (4*math.pi**2*1e10*3e8*gvd[np.argmin(np.abs(lam-405.))])
	plt.plot(lam,gvd,'k-')
	plt.show()
	sys.exit()
	PlotMaterial("GaN",2.266,2.13e-2,3.85e-3,1.5,3.1,'k')
	PlotMaterial("AlN",2.19,2.72e-3,0,1.5,3.1,'b')	
	PlotSellmeier("Ref",5.15,0.35,339.8,1.5,3.1,'k')
	PlotSellmeier("Ref",1.0,3.12,138,1.5,3.1,'b')
	
	plt.legend(loc='upper left',prop={'size':24})
	#plt.xlim([2.45,2.95])
	#plt.ylim([2500,11000])
	plt.xlabel('Energy (eV)')	
	plt.ylabel('$n$')
	plt.grid()
	ax2 = ax1.twiny()
	ax2.set_xlim(ax1.get_xlim())
	new_tick_locations 	= 1240. / np.linspace(800,400,5)
	ax2.set_xticks(new_tick_locations)
	ax2.set_xticklabels(tick_function(new_tick_locations))
	ax2.set_xlabel("Wavelength (nm)")	
	plt.savefig('A3641-dispersion.png',bbox_inches='tight')
	plt.savefig('A3641-dispersion.pdf',bbox_inches='tight')	
	plt.show()