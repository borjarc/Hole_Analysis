import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as spop
import scipy.stats as stat
import sys
import math
import numpy as np
from NanobeamUPL import *
from ParseFiles import *
from uPL import *
import dill
import os

def ParseFilename(filen,delim="_"):
	eles 	= filen.split(delim)
	i 		= 0
	row 	= int(eles[i+1])
	col 	= int(eles[i+2])
	pwr 	= int(eles[i+3].split("uW")[0])
	try:
		if len(eles) > 4:
			num 	= int(eles[-1].split(".txt")[0])
		else:
			num 	= 0
	except:
		num = 10
	return [row, col, pwr, num]
	
def AnalyzeDirectory(dirname, sID, delim="_", sw=None, grat=None, tint=None, FP=True):
	nbupl 			= []	
	onlyfiles 		= [f for f in listdir(dirname) if isfile(join(dirname, f))]	
	onlyfiles 		= filter(lambda x: sID in x, onlyfiles)
	IDS 			= np.array([])
	dirout 			= sID 
	if not os.path.exists(dirout):
		os.makedirs(dirout)
	for f in onlyfiles:
		[row, col, pwr, num] 	= ParseFilename(f,delim)
		print "%s (%d,%d) #%d" % (f, row, col, num)
		if sw is None:
			up1 		= LabSpec6("%s/%s" % (dirname,f),"(%d,%d) %d" % (row,col,num))
		else:
			up1 		= uPL("%s/%s" % (dirname,f),"(%d,%d) %d" % (row,col,num),sw,tint,grat)
		figname 	= "%s/%s.png" % (dirout,f.split(".txt")[0])
		peaks 		= IdentifyPeaks(up1,SAVEFIG=figname,FP=True)
		if peaks is None:
			continue
		print "\n"
		nbptmp 		= NanobeamUPL("NB-N-%d" % num, sID, row, col)
		if nbupl is []:
			nbptmp.AddUPL(f,peaks)
			print "\tAdding new NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())
			nbupl.append(nbptmp)
			IDS 	= np.hstack([IDS, 100*row+col])
		else:
			myID 	= 100*row+col
			if np.in1d(myID, IDS)[0]:
				i 	= np.argmin(np.abs(IDS-myID))
				print "\tAppending to NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())
				nbupl[i].AddUPL(f,peaks)
				print nbupl[i].GetAll().keys()
			else:
				nbptmp.AddUPL(f,peaks)
				print "\tAdding to NanobeamUPL %s %s (%d,%d)" % (nbptmp.GetSampleID(), nbptmp.GetCellID(), nbptmp.GetRow(), nbptmp.GetColumn())			
				nbupl.append(nbptmp)	
				IDS 	= np.hstack([IDS,myID])
	return nbupl			

	
def FitPickleDir(dirname, sID, outpkl):
	fits			= AnalyzeDirectory(dirname,sID,delim="_")
	dill.dump(fits,open("%s.pkl" % (outpkl),'wb'))	
	
def FitPickleDir2(dirname, sID, outpkl, sw, grat, tint):
	fits			= AnalyzeDirectory(dirname,sID,delim="_",sw=sw,grat=grat,tint=tint)
	dill.dump(fits,open("%s.pkl" % (outpkl),'wb'))		
	
if __name__=="__main__":
	dir1 			= "U:/Spectroscopy/QOLab/20170215-A3531+4-DegradationTest/"
	onlyfiles 		= [f for f in listdir(dir1) if isfile(join(dir1, f))]
	degrade			= AnalyzeDirectory(dir1,onlyfiles,"Q445_m1",delim="-")
	
	dill.dump(quanT,open('A3531+4_Quan_seq.pkl','wb'))		
	#dill.dump(quanD,open('A3531+4_Quan_dwn.pkl','wb'))			