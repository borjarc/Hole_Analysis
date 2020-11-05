from numpy import * 
from math import floor
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join, exists
import sys 
from uPL import * 
from NanobeamUPL import *

def ExtractVitals(lst, FILTER=None):
	'''
	Extract data from a list of uPL objects, assuming that that there are maximally two resonances in each.
	Average multiple measurements into one set of fits.
	'''
	lamB 	= []
	QB 		= []
	AB 		= []
	lamR 	= []
	QR		= []
	AR		= []
	lSplit 	= []
	for nb in lst:
		#Hold 
		lam1 		= []
		fwhm1 		= []
		A1 			= []
		wt1 		= []	
		Nfit 		= len(nb.GetAllFits())
		if FILTER == 'H' and (nb.GetColumn() == 3 or nb.GetColumn() == 4):
			continue
		elif FILTER == 'V' and (nb.GetColumn() == 1 or nb.GetColumn() == 2):
			continue
		#Go over multiple files
		for fits in nb.GetAllFits():
			#Have temporary variables
			lams 		= np.array(map(lambda x: x.GetLambda0(), fits))
			fwhm 		= np.array(map(lambda x: x.GetFWHM(), fits))		
			A 			= np.array(map(lambda x: x.GetA(),fits))
			if len(fits) > 2:
				print "\t%s %s (%d,%d)" % (nb.GetSampleID(), nb.GetCellID(), nb.GetRow(), nb.GetColumn())
				for i in range(lams.shape[0]):
					print "\t\tlam: %0.2f nm Q: %d A: %0.1f" % (lams[i], lams[i]/fwhm[i], A[i])
			#Check for duplicate measurements
			for i in range(lams.shape[0]):
				lami 	= lams[i]
				fwhmi 	= fwhm[i]
				ADDME 	= True
				for j in range(len(lam1)):
					lamj 	= lam1[j]
					fwhmj 	= fwhm1[j]
					dlam 	= np.abs(lamj-lami)
					sig 	= 0.5*np.max([fwhmi,fwhmj])
					if dlam < sig:
						ADDME 	= False 
						break						
				if ADDME:
					#New wavelength
					lam1.append(lami)
					fwhm1.append(fwhmi)
					A1.append(A[i])
					wt1.append(1)
				else:
					#Average into the pre-existing array
					lamtmp 	= np.array(lam1)
					ii 		= np.argmin(np.abs(lamtmp - lami))
					lam1[ii] 	= (wt1[ii]*lam1[ii]+lami) / (wt1[ii]+1)
					fwhm1[ii] 	= (wt1[ii]*fwhm1[ii]+fwhmi) / (wt1[ii]+1)
					A1[ii] 		= (wt1[ii]*A1[ii]+A[i]) / (wt1[ii]+1)
					wt1[ii] 	= wt1[ii] + 1
					print "\t\tAveraging %s %s (%d,%d) {l=%0.2f nm, Q=%d} to {l=%0.2f nm, Q=%d}" % (nb.GetSampleID(), nb.GetCellID(), \
					nb.GetRow(), nb.GetColumn(), lami, lami/fwhmi, lam1[ii], lam1[ii]/fwhm1[ii])					
		#Now, add the temporary arrays into the regular arrays 
		lam1 	= np.array(lam1)
		fwhm1 	= np.array(fwhm1)
		A1 		= np.array(A1)
		wt1 	= np.array(wt1)
		if lam1.shape[0] > 2:
			ii 		= np.argsort(A1)
			lam1	= lam1[ii]
			fwhm1 	= fwhm1[ii]
			A1 		= A1[ii]
			wt1 	= wt1[ii]
			lam1 	= lam1[-2:]
			fwhm1	= fwhm1[-2:]
			A1 		= A1[-2:]
			wt1 	= wt1[-2:]
			print "\t\tKeeping %0.1f and %0.1f nm" % (lam1[0],lam1[1])
		if lam1.shape[0] == 1:
			AB.append(A1[0])
			lamB.append(lam1[0])
			QB.append(lam1[0]/fwhm1[0])
		elif lam1.shape[0] == 2:		
			ii 	= np.argsort(lam1)
			lam1 	= np.sort(lam1)
			fwhm1 	= fwhm1[ii]
			A1 		= A1[ii]
			wt1 	= wt1[ii]
			AB.append(A1[0])
			lamB.append(lam1[0])
			QB.append(lam1[0]/fwhm1[0])
			AR.append(A1[1])
			lamR.append(lam1[1])
			QR.append(lam1[1]/fwhm1[1])	
			lSplit.append(lam1[1]-lam1[0])
		else:
			print "Found %d peaks for %s %s (%d,%d)" % (lam1.shape[0], nb.GetSampleID(), nb.GetCellID(), nb.GetRow(), nb.GetColumn())
	return [np.array(AB),np.array(lamB),np.array(QB),np.array(AR),np.array(lamR),np.array(QR),np.array(lSplit)]

def ParseFilename(filen):
	eles 	= filen.split("_")
	cID 	= eles[0]
	loc 	= int(eles[1])
	row 	= int(floor(loc/10))
	col 	= loc % 10 
	sw 		= int(eles[2].split("um")[0])
	tint 	= int(eles[4].split("s")[0])
	if len(eles) == 6:
		grat 	= int(eles[5].split(".dat")[0])
		num 	= 1
	else:
		grat 	= int(eles[5])
		num 	= int(eles[6].split(".dat")[0])
	return [cID, row, col, sw, tint, grat, num]
	
def ParseFilename021616(filen):
	eles 	= filen.split("_")
	cID 	= eles[0]
	loc 	= int(eles[1])
	row 	= int(floor(loc/10))
	col 	= loc % 10 
	sw 		= int(eles[4].split("um")[0])
	tint 	= int(eles[3].split("s")[0])
	if len(eles) == 6:
		grat 	= int(eles[5].split(".dat")[0])
		num 	= 1
	else:
		grat 	= int(eles[5])
		num 	= int(eles[6].split(".dat")[0])
	return [cID, row, col, sw, tint, grat, num]
	
def FitCell(sID, dirname, READ0216=False):
	#Go through a folder and fit all the files, putting multiple spectra into the same NanobeamUPL object.
	onlyfiles 	= [f for f in listdir(dirname) if isfile(join(dirname, f))]
	nbupl 		= []
	for f in onlyfiles:
		if READ0216:
			[cID, row, col, sw, tint, grat, num] 	= ParseFilename021616(f)
		else:
			[cID, row, col, sw, tint, grat, num] 	= ParseFilename(f)
		print "Fitting A3405C %s (%d,%d) %d um %d s %d lp/mm #%d" % (cID, row, col, sw, tint, grat, num)
		up1 	= uPL("%s/%s" % (dirname, f), "1.6 kW/cm$^2$", sw/1000., tint, grat)	
		figname 	= "%s.png" % (f.split(".dat")[0])
		peaks 		= IdentifyPeaks(up1,SAVEFIG=figname)
		if peaks is None:
			continue
		print "\n"
		nbptmp 		= NanobeamUPL(sID, cID, row, col)
		IDS 		= np.array([map(lambda x: 10*x.GetRow()+x.GetColumn(), nbupl)])
		if nbupl is []:
			nbptmp.AddUPL(f,peaks)
			print "\tAdding new NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())
			nbupl.append(nbptmp)
		else:
			myID 	= 10*row + col 
			if np.in1d(np.array([myID]), IDS)[0]:
				i 	= np.argmin(np.abs(IDS-myID))
				print "\tAppending to NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())
				nbupl[i].AddUPL(f,peaks)
				print nbupl[i].GetAll().keys()
			else:
				nbptmp.AddUPL(f,peaks)
				print "\tAdding to NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())			
				nbupl.append(nbptmp)		
	return nbupl
		
if __name__=="__main__":

	#assert(len(sys.argv) == 2)
	#assert(exists(sys.argv[1]))
	sID 		= "A3405C"
	dirname 	= "U:/Spectroscopy/uPL/20160222-A3405C/D7"
	FitCell(sID,dirname)