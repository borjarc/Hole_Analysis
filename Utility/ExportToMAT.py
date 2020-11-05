from scipy.io import savemat
from AnalyzeDirectory import *
import dill
import numpy as np
from uPL import *

def ExportToMAT(filen,prefix,infile):
	cell 		= dill.load(open(infile,'rb'))
	Nrow 	= len(cell)
	Ncol 	= 0
	k 		= 0 
	mylam 	= np.zeros((Nrow,10))
	myq 	= np.zeros((Nrow,10))		
	#First filter out duplicate fits and get the size of the array 
	for nb in cell:
		fits 	= nb.GetAllFits()
		lam 	= np.array(map(lambda x: x.GetLambda0(), fits))
		fwhm	= np.array(map(lambda x: x.GetFWHM(), fits))
		sig 	= fwhm / (2*math.sqrt(2*np.log(2)))
		dlam 	= np.outer(np.ones(lam.shape),lam) - np.outer(lam,np.ones(lam.shape))
		sig 	= np.outer(np.ones(sig.shape),sig)
		ri, ci 	= np.nonzero(np.abs(dlam) < 0.5*sig)
		badi 	= []
		oki 	= []
		for i in range(ri.shape[0]):
			if (ri[i] == ci[i]) and (ri[i] not in badi):
				oki.append(ri[i])
			else:
				badi.append(ci[i])
		count 	 = 0
		for j in range(len(oki)):
			qj 			= lam[j]/(fwhm[j]-0.03) #IRF -> old setup
			if qj < 4000:
				mylam[k,j] 	= lam[j]
				myq[k,j] 	= qj 
				count 	= count + 1 
		if count > Ncol:
			Ncol 	= count
		k 	= k + 1
	nameLam 	= "%s_lam" % (prefix)
	nameQ 		= "%s_Q" % (prefix)
	mylam 	= mylam[:,:Ncol]
	myq 	= myq[:,:Ncol]
	savemat(filen, {nameLam:mylam, nameQ:myq})
	
if __name__=="__main__":
	#ExportToMAT("U:/Analysis/20170225-A3531+3-CO2/CO2-Fits.mat", "CO2", "U:/Analysis/20170225-A3531+3-CO2/CO2-Fits.pkl")
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-133-Fits.mat", "w133", "U:/Analysis/20170216-A3530-1/W1-133-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-144-Fits.mat", "w144", "U:/Analysis/20170216-A3530-1/W1-144-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-150-Fits.mat", "w150", "U:/Analysis/20170216-A3530-1/W1-150-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-155-Fits.mat", "w155", "U:/Analysis/20170216-A3530-1/W1-155-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-160-Fits.mat", "w160", "U:/Analysis/20170216-A3530-1/W1-160-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-166-Fits.mat", "w166", "U:/Analysis/20170216-A3530-1/W1-166-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-172-Fits.mat", "w172", "U:/Analysis/20170216-A3530-1/W1-172-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-177-Fits.mat", "w177", "U:/Analysis/20170216-A3530-1/W1-177-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-183-Fits.mat", "w183", "U:/Analysis/20170216-A3530-1/W1-183-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-188-Fits.mat", "w188", "U:/Analysis/20170216-A3530-1/W1-188-Fits.pkl")	
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-194-Fits.mat", "w194", "U:/Analysis/20170216-A3530-1/W1-194-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/W1-200-Fits.mat", "w200", "U:/Analysis/20170216-A3530-1/W1-200-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L421-Fits.mat", "l421", "U:/Analysis/20170216-A3530-1/L421-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L431-Fits.mat", "l431", "U:/Analysis/20170216-A3530-1/L431-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L440-Fits.mat", "l440", "U:/Analysis/20170216-A3530-1/L440-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L450-Fits.mat", "l450", "U:/Analysis/20170216-A3530-1/L450-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L460-Fits.mat", "l460", "U:/Analysis/20170216-A3530-1/L460-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L469-Fits.mat", "l469", "U:/Analysis/20170216-A3530-1/L469-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L478-Fits.mat", "l478", "U:/Analysis/20170216-A3530-1/L478-Fits.pkl")		
	ExportToMAT("U:/Analysis/20170216-A3530-1/L489-Fits.mat", "l489", "U:/Analysis/20170216-A3530-1/L489-Fits.pkl")			