import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
import os.path 
import sys
from uPL import *
from matplotlib.widgets import Slider, Button, RadioButtons, Cursor

def PlotIntensity(obj):
	xc 		= obj.GetX()
	yc 		= obj.GetY()
	xu 		= np.sort(np.unique(xc))
	yu 		= np.sort(np.unique(yc))
	intens 	= obj.GetIntensity().reshape(xu.shape[0],yu.shape[0])
	intens 	= intens/np.max(np.max(intens))
	cm 		= plt.contourf(yu,xu,intens,30)
	plt.gca().set_aspect('equal')	
	cbar 	= plt.colorbar(cm)
	cbar.set_ticks(np.linspace(0,1,11))	
	plt.xlabel('x ($\mu$m)')
	plt.ylabel('y ($\mu$m)')	
	cbar.ax.set_ylabel('PL Intensity (arb. units)')	

def PlotMapping(obj):
	fig, ax = plt.subplots()
	#plt.subplots_adjust(left=0.25, bottom=0.25)
	
	xc 		= obj.GetX()
	yc 		= obj.GetY()
	xu 		= np.unique(xc)
	yu 		= np.unique(yc)	
		
	wl 		= obj.GetWavelength()
	axfreq 	= plt.axes([0.25, 0.25, 0.55, 0.03])	
	mywl 	= Slider(axfreq, "WL (nm)", np.min(wl), np.max(wl), valinit=0.5*(np.min(wl)+np.max(wl)), valfmt='%1.2f')
	ii 		= np.argmin(np.abs(wl-mywl.val))
	
	intens 	= obj.GetMonochromatic(wl[ii])
	intens 	= (intens - np.min(intens)) / np.max(intens - np.min(intens))
	intens 	= intens.reshape((yu.shape[0],xu.shape[0])).transpose()
	wl0 	= wl[ii]
	ax1 	= plt.subplot(311)
	cm 	 	= ax1.contourf(yu,xu,intens,30,cmap=plt.cm.jet)	
	ax1.set_aspect('equal')	
	ax1.set_xlabel('x ($\mu$m)')
	ax1.set_ylabel('y ($\mu$m)')
	cbar 	= ax1.figure.colorbar(cm)
	cbar.ax.set_ylabel('PL Intensity (arb)')

	ax2 	= plt.subplot(312)
	#Set up the position sliders
	axX 	= plt.axes([0.25, 0.20, 0.55, 0.03])	
	myX 	= Slider(axX, "y (um)", np.min(xu), np.max(xu), valinit=0.5*(np.min(xu)+np.max(xu)), valfmt='%1.2f')
	axY 	= plt.axes([0.25, 0.15, 0.55, 0.03])	
	myY 	= Slider(axY, "x (um)", np.min(yu), np.max(yu), valinit=0.5*(np.min(yu)+np.max(yu)), valfmt='%1.2f')
	c1 		= plt.Circle((myY.val,myX.val),0.25,color='w')
	ax1.add_artist(c1)	
	
	ix 			= np.argmin(np.abs(xu-myX.val))
	iy 			= np.argmin(np.abs(yu-myY.val))
	speci 		= obj.GetSingleSpectrum(xu[ix],yu[iy])
	lines 		= ax2.plot(wl, speci, 'k-', lw=1)
	lines 		= ax2.plot(wl0*np.ones(10),np.linspace(0.95*np.min(speci),1.05*np.max(speci),10),'--',color='#FF69B4',lw=4)
	ax2.set_xlabel('Wavelength (nm)')
	ax2.set_ylabel('PL counts (arb units)')
	ax2.set_xlim([np.min(wl),np.max(wl)])
	ax2.set_ylim([0.95*np.min(speci),1.05*np.max(speci)])
	ax2.grid(True)

		
	def update2(val):
		ix 			= np.argmin(np.abs(xu-myX.val))
		iy 			= np.argmin(np.abs(yu-myY.val))
		c1.center 	= yu[iy], xu[ix]
		speci 		= obj.GetSingleSpectrum(xu[ix],yu[iy])
		ax2.set_xlim([np.min(wl),np.max(wl)])
		ax2.set_ylim([0.95*np.min(speci),1.05*np.max(speci)])
		ax2.lines.pop()
		ax2.lines.pop()
		lines 		= ax2.plot(wl, speci, 'k-', lw=1)	
		ii 			= np.argmin(np.abs(wl-mywl.val))
		lines 		= ax2.plot(wl[ii]*np.ones(10),np.linspace(0.95*np.min(speci),1.05*np.max(speci),10),'--',color='#FF69B4',lw=4)		
		fig.canvas.draw()	
	def update(val):
		ii 		= np.argmin(np.abs(wl-mywl.val))
		intens 	= obj.GetMonochromatic(wl[ii])
		intens 	= (intens - np.min(intens)) / np.max(intens - np.min(intens))
		intens 	= intens.reshape((xu.shape[0],yu.shape[0]))
		cm 		= ax1.contourf(yu,xu,intens,30,cmap=plt.cm.jet) 
		ax1.set_aspect('equal')
		update2(val)
		fig.canvas.draw()
		
	mywl.on_changed(update)		
	myX.on_changed(update2)
	myY.on_changed(update2)
	plt.tight_layout()

if __name__=="__main__":
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[8,22*2.0/(sqrt(5)+1)]}	
	matplotlib.rcParams.update(params)
	
	f1 		= "U:/Spectroscopy/QOLab/20170725-A3572-1-SiO2-5K-Anneal/mapscan_1_15um.csv"	
	ms 		= HS_Mapscan(f1,100,"hworld")
	PlotMapping(ms)
	plt.show()