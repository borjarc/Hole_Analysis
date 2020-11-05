from NanobeamHoles import * 
from SEMImage import *
import cPickle as pickle
import copy
import os.path
import matplotlib as mpb
import matplotlib.pyplot as plt
import scipy.stats as stat
from math import sqrt

def ExtractData(dat, r0, w0=None):
	if w0 is None:
		nlst 	= filter(lambda x: x.GetR0() == r0, dat)	
	else:
		nlst 	= filter(lambda x: x.GetR0() == r0 and x.GetW0() == w0, dat)

	rlst 	= np.array([])
	yoff 	= np.array([])
	for nbi in nlst:
		rlst 		= np.hstack([rlst,nbi.GetR()])
		x 			= nbi.GetX0()
		y 			= nbi.GetY0()
		p 			= np.polyfit(x,y,1)
		xmn 		= (x*(1+p[0])-p[0]*p[1])/(1+p[0]**2)
		ymn 		= np.polyval(p, xmn)	
		yoff 		= np.hstack([yoff,y-ymn])		

	
	#Filter for outliers
	Nr0 		= rlst.shape[0]
	Nr 			= rlst.shape[0] + 1
	ii 			= np.arange(rlst.shape[0])
	while Nr > rlst.shape[0] and ii.shape[0] > 1:
		q75, q25 	= np.percentile(rlst, [75 ,25])	
		iqr 		= q75 - q25
		ii	 		= np.nonzero(np.logical_and(rlst<q75+1.5*iqr,rlst>q25-1.5*iqr))[0]
		rlst 		= rlst[ii]
		yoff 		= yoff[ii]
		Nr 			= ii.shape[0]
	if ii.shape[0] == 1:
		return [None, None]	
	'''
	if w0 is None:
		print "R0: %d %d holes %d outliers" % (r0+11, rlst.shape[0], Nr0-rlst.shape[0])	
	else:
		print "R0: %d W0: %d %d holes %d outliers" % (r0+11, w0, rlst.shape[0], Nr0-rlst.shape[0])
	#print "R0: %0.1f +/- %0.1f nm" % (np.mean(rlst),np.std(rlst))
	'''
	return [rlst,yoff]

def PlotHistogram(dat,lab=None,NBIN=None,color='k',off=0):
	bin0 		= np.mean(dat)
	binstd 		= np.std(dat)
	if NBIN is None:
		hist, bin 	= np.histogram(dat)
	else:
		bin 		= np.linspace(bin0-2*binstd,bin0+2*binstd,NBIN)
		hist, bin 	= np.histogram(dat, bins=bin)
	width 		= 0.9 * (bin[1] - bin[0])
	ctr 		= 0.5*(bin[1:] + bin[:-1])
	if lab is not None:
		plt.bar(ctr+off*width, 100*hist/np.sum(hist), align='center', width=width, color=color, label=lab)	
	else:
		plt.bar(ctr+off*width, 100*hist/np.sum(hist), align='center', width=width, color=color)

if __name__=="__main__":
	r0 		= np.arange(30,50,5) - 11
	'''
	params = {'font.family' : 'serif',
			'text.fontsize': 18, 'axes.labelsize':16, 
			'axes.titlesize':18,'xtick.labelsize':18,
			'ytick.labelsize':18,
			'text.usetex':True, 'text.latex.unicode':True,
			'figure.figsize':[6,6*2.0/(sqrt(5)+1)],
			'figure.autolayout':True}
	'''
	params 	= {'font.family' : 'serif', "font.serif": ['Times New Roman'] ,'font.size' : 24,
	'figure.autolayout':True, 'figure.figsize':[8,8*2.0/(sqrt(5)+1)]}	
	mpb.rcParams.update(params)
	dat 	= pickle.load(open('A3406-EtchDecember-2.pkl','rb'))
	w0 		= [200, 230]
	i 		= 0
	#Check to see if there is a difference between the two widths
	cols		= plt.cm.Reds(np.linspace(0, 1, 5))		
	cols 		= cols[1: ]	
	cols 		= cols[-1::-1]
	for ri in r0:
		
		[rvec1,y0vec1] 	= ExtractData(dat,ri,200)
		[rvec2,y0vec2] 	= ExtractData(dat,ri,230)
		_,pr1 			= stat.normaltest(rvec1)
		_,pr2 			= stat.normaltest(rvec2)
		_,py1 			= stat.normaltest(y0vec1)
		_,py2 			= stat.normaltest(y0vec2)		
		_,pbR			= stat.bartlett(rvec1, rvec2)
		_,pbY 			= stat.bartlett(y0vec1, y0vec2)
		_, pr 			= stat.ttest_ind(rvec1, rvec2, equal_var=False)
		_, py 			= stat.ttest_ind(y0vec1, y0vec2, equal_var=False)
		print "r: %d nm" % (ri+11)
		print "w: 200 nm N=%d \tr = %0.1f +/- %0.2f nm y0 = %0.1f +/- %0.2f nm" % (rvec1.shape[0],np.mean(rvec1),np.std(rvec1),\
		np.mean(y0vec1),np.std(y0vec1))
		print "w: 230 nm N=%d\tr = %0.1f +/- %0.2f nm y0 = %0.1f +/- %0.2f nm" % (rvec2.shape[0],np.mean(rvec2),np.std(rvec2),\
		np.mean(y0vec2),np.std(y0vec2))		
		rvec 			= np.hstack([rvec1,rvec2])
		y0vec 			= np.hstack([y0vec1,y0vec2])
		print "Total N=%d\tr = %0.1f +/- %0.2f nm y0 = %0.1f +/- %0.2f nm" % (rvec.shape[0],np.mean(rvec),np.std(rvec),\
		np.mean(y0vec),np.std(y0vec))			
		print "\tIs normal? p_r1: %0.2f p_r2: %0.2f p_y1: %0.2f p_y2: %0.2f" % (pr1,pr2,py1,py2)
		print "\tHas same variance? p_r: %0.3f p_y: %0.3f" % (pbR, pbY)
		print "\tHas same mean? p_r: %0.3f p_y: %0.3f" % (pr, py)
		print ""
		PlotHistogram(np.hstack([y0vec1,y0vec2])+i*4, lab='%d nm'%(2*np.mean(rvec1)),color=cols[i%len(cols)])	
		i 				= i + 1
	plt.xlabel('Offset [nm]')
	plt.ylabel('Frequency [%]')
	plt.grid()
	plt.savefig('20160501-HoleLocationHistogram.pdf',bbox_inches='tight')
	plt.savefig('20160501-HoleLocationHistogram.png',bbox_inches='tight')
	plt.clf()	
	cols		= plt.cm.Blues(np.linspace(0, 1, 5))		
	cols 		= cols[1: ]	
	cols 		= cols[-1::-1]	
	print cols[2]*255
	[rvec1,y0vec1] 	= ExtractData(dat,30-11,200)
	[rvec2,y0vec2] 	= ExtractData(dat,30-11,230)	
	print "d: 50 N: %d r0: %0.1f +/- %0.2f" % (rvec1.shape[0],np.mean(rvec1),np.std(rvec1))			
	PlotHistogram(rvec1*2, lab='%d nm'%(2*np.mean(rvec1)),color=cols[0])	
	[rvec1,y0vec1] 	= ExtractData(dat,35-11)
	[rvec2,y0vec2] 	= ExtractData(dat,35-11,230)	
	print "d: 60 N: %d r0: %0.1f +/- %0.2f" % (rvec1.shape[0],np.mean(rvec1),np.std(rvec1))
	PlotHistogram(np.hstack([rvec2,rvec1])*2, lab='%d nm'%(2*np.mean(rvec1)),color=cols[1])	
	[rvec1,y0vec1] 	= ExtractData(dat,40-11)
	[rvec2,y0vec2] 	= ExtractData(dat,40-11,230)	
	print "d: 70 N: %d r0: %0.1f +/- %0.2f" % (rvec1.shape[0],np.mean(rvec1),np.std(rvec1))	
	PlotHistogram(np.hstack([rvec2,rvec1])*2, lab='%d nm'%(2*np.mean(rvec1)),color=cols[2])	
	[rvec1,y0vec1] 	= ExtractData(dat,45-11,200)
	[rvec2,y0vec2] 	= ExtractData(dat,45-11,230)	
	PlotHistogram(rvec1*2, lab='%d nm'%(2*np.mean(rvec1)),color=cols[3])
	print "d: 80 N: %d r0: %0.1f +/- %0.2f" % (rvec1.shape[0],np.mean(rvec1),np.std(rvec1))					
	plt.xlabel('Diameter [nm]')
	plt.ylabel('Frequency [%]')
	plt.grid()
	plt.savefig('20160501-HoleDiameterHistogram.pdf',bbox_inches='tight')
	plt.savefig('20160501-HoleDiameterHistogram.png',bbox_inches='tight')	
	plt.clf()
	
	
