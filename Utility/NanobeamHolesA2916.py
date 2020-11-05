import sys
import cv2
import os.path
import math 
import numpy as np
import matplotlib.pyplot as plt
from SEMImage import *
import matplotlib.cm as cm
import scipy.optimize as spop

class NanobeamHoles:

	def __init__(self,sem,r,rtol,beamw,beamtol):
		assert isinstance(sem,ZeissSEMImage)
		self.sem 	= sem
		[x0,y0,r0,resid0,w0,wstd0] = self.__AnalyzeImage__(r,rtol,beamw,beamtol,True)
		[x1,y1,r1,resid1,w1,wstd1] = self.__AnalyzeImage__(r,rtol,beamw,beamtol,False)
		if r0.shape[0] >= r1.shape[0] :
			self.EQHIST 		= True
		else:
			self.EQHIST 		= False	
		print self.EQHIST
		[self.x0,self.y0,self.r,self.resid,self.w,self.wstd] = self.__AnalyzeImage__(r,rtol,beamw,beamtol,self.EQHIST)
		
	def GetX0(self):
		return self.x0 
		
	def GetY0(self):
		return self.y0 
		
	def GetR(self):
		return self.r 
		
	def GetResid(self):
		return self.resid 
		
	def GetBeamWidth(self):
		return self.w 
		
	def GetBeamWidthStd(self):
		return self.wstd
		
	def GetEqHist(self):
		return self.EQHIST
		
	def __FindCorners__(self,px,dat,bw,btol,SMOOTH,DEBUG=False):
		#Find the nanobeam corners of a single cross-section of the image
		smth 		= np.convolve(np.ones(SMOOTH),dat,mode='same') / SMOOTH 
		mxV 		= np.argsort(smth)[-1::-1]
		if mxV[0] >= mxV[1]:
			hi 	= mxV[0]
			lo 	= mxV[1] 
		else:
			hi 	= mxV[1] 
			lo 	= mxV[0]

		i 	= 2
		while px * (hi - lo) < bw*(1 - btol) and i < mxV.shape[0]:
			if mxV[i] > hi:
				hi 	= mxV[i]
			else:
				lo 	= mxV[i]
			i 	= i + 1
		
		if DEBUG:
			plt.plot(px*np.arange(dat.shape[0]),dat,'k-',lw=1)
			plt.plot(px*np.arange(smth.shape[0]),smth,'b-',lw=1)
			plt.plot(px*lo,smth[lo],'ro',ms=8)
			plt.plot(px*hi,smth[hi],'ro',ms=8)
			plt.xlabel('Distance [nm]')
			plt.ylabel('Intensity [arb]')
			plt.show()
		return [lo,hi]
		
	def __FindEdges__(self,img,bot,top,bw,btol,DEBUG=True): 
		#Maximize the average integral of intensity over the given line
		img0 	= img.copy()
		img 		= 255 - img
		def FitFxn(p,img2):
			xc 		= np.arange(img2.shape[1])		
			m 	= p[0]
			b 	= p[1]
			yc 	= np.round(m*xc+b,0).astype('int')
			ii 		= np.logical_and( yc >= 0, yc < img.shape[0] )	
			if np.sum(ii) > 0:
				return np.sum(img2[yc[ii],xc[ii]] )
			return 1e6
		#Blank out the top or bottom half of the image
		imgT 	= img.copy()
		imgB 	= img.copy()
		imgT[:(bot+top)/2,:] 	= 255
		imgT[int((bot+top)/2+0.5*bw*(1+btol)):,:] 	= 255
		imgB[(bot+top)/2:,:] 	= 255
		imgB[:int((bot+top)/2-0.5*bw*(1+btol)),:] 	= 255	
		#Brute force fit
		p2 		= spop.brute(lambda x: FitFxn(x,imgT),(( -10*math.pi/180,10*math.pi/180 ), ((bot+top)/2,(bot+top)/2+0.5*bw*(1+btol)) ))
		p1 		= spop.brute(lambda x: FitFxn(x,imgB),(( -10*math.pi/180,10*math.pi/180 ), ((bot+top)/2-0.5*bw*(1+btol),(bot+top)/2) ))
		#Shift the lines vertically until they are on the maximum |gradient|	
		vt0 		= FitFxn(p2,imgT)
		vt1 		= FitFxn([p2[0],p2[1]+1],imgT)
		d0 		= 0
		ii 		= 1
		while vt1 - vt0 > d0 and ii < imgT.shape[0]:
			ii 	= ii + 1
			d0 	= vt1 - vt0
			vt0 	= vt1 
			vt1 	= FitFxn([p2[0],p2[1]+ii],imgT)
		if ii == imgT.shape[0]:
			ii 	= 0 
		p2[1] 	= p2[1] + ii
		#The lower line 
		vb0 		= FitFxn(p1,imgB)
		vb1 		= FitFxn([p1[0],p1[1]-1],imgB)
		d0 		= 0
		ii 		= -1
		while vb1 - vb0 > d0 and ii < p1[1]:
			ii 	= ii - 1
			d0 	= vb1 - vb0
			vb0 	= vb1 
			vb1 	= FitFxn([p1[0],p1[1]+ii],imgB)
		if ii >= p1[1]:
			ii 	= 0
		p1[1] 	= p1[1] + ii		
		#Now, fit a line for the hole locations 
		#p30 		= [0, np.argmin( np.sum(img0[bot:top],1) ) + bot]
		y0 		= int(np.round(np.polyval(p1,img.shape[1]/2),0))
		y1 		= int(np.round(np.polyval(p2,img.shape[1]/2),0))
		imgM 	= img0.copy()
		imgM[:y0,:] 	= 255
		imgM[y1:,:] 	= 255
		xc 			= np.arange(imgM.shape[1])
		def FitFxn2(p):
			yc 		= np.round(np.polyval(p,xc),0).astype('uint16')
			return np.sum(imgM[yc,xc])
		p3 		= spop.brute(FitFxn2,(( -1*math.pi/180,1*math.pi/180 ), (p1[1],p2[1])))
		#if ier < 1 or ier > 4:
		#	print "Hole centerline fit failed.  Reverting to average of beam edges."
		#	p3 		= p30
		if DEBUG:
			xc 		= np.arange(img0.shape[1])
			plt.imshow(img0,cmap = cm.Greys_r)
			plt.plot(xc, np.polyval(p2,xc),'m-',lw=2)
			plt.plot(xc, np.polyval(p1,xc),'m-',lw=2)			
			plt.plot(xc,np.polyval(p3,xc),'w-',lw=2)
			plt.show()			
		return [p1,p2,p3]
		
	def __FindUnitCells__(self,x,y,r,rtol,DEBUG=False):
		mxV 		= np.argsort(y)[-1::-1]
		mxX 		= [mxV[0]] 

		#Local function to test whether candidate value is within r*(1-rtol) of another element
		def CanAdd(dstVec, trial):
			for i in dstVec:
				if np.abs(i - trial) < 2*r*(1-rtol):
					return False 
			return True
		i 			= 1
		while i < x.shape[0] and y[mxV[i]] > np.mean(y):
			if CanAdd(mxX, mxV[i]):
				mxX.append(mxV[i])
			i 		= i + 1 
		mxX 	= np.sort(np.array(mxX))
		if DEBUG:
			plt.plot(x,y,'k-',lw=2)
			plt.plot(mxX,y[mxX],'ro',ms=8)
			plt.show()
		#Now calculate where the dividing line between unit cells is located
		abnd 	= [mxX[0]] 
		ybar 	= np.mean(y)
		for ii in range(1, mxX.shape[0]):
			if np.any( y[mxX[ii-1]:mxX[ii]+1] < ybar):
				abnd.append(mxX[ii])
			else:
				abnd.append(int(np.round(0.5*(mxX[ii-1]+mxX[ii]))))
		abnd 	= np.array(abnd)
		y1mn 	= np.min(y[:abnd[1]])
		if y1mn < ybar:
			r0 	= [np.argmin(y[:abnd[1]])]
		else:
			r0 	= []
		for ii in range(1,abnd.shape[0]):
			y1mn 	= np.min(y[abnd[ii-1]:abnd[ii]])
			if y1mn < ybar and CanAdd(r0, np.round(0.5*(abnd[ii-1]+abnd[ii]),0)):
				r0.append( np.round(0.5*(abnd[ii-1]+abnd[ii]),0) )
		y1mn 	= np.min(y[abnd[-1]:])
		if y1mn < ybar and CanAdd(r0, np.round(abnd[-1]+np.argmin(y[abnd[-1]:]),0)):
			r0.append( abnd[-1]+np.argmin(y[abnd[-1]:]) )
		return np.array(r0).astype('uint16')

		
	def __AnalyzeImage__(self,r,rtol,beamw,beamtol,EQHIST,DEBUG=False,SMOOTH=20):
		'''
		Find the nanobeam holes in the image
		'''
		img  			= self.sem.GetImageCropped()

		#Extract the nanobeam orientation
		px 				= self.sem.GetPixelSize()
		if px > 2:
			px 			= 0.1*px
		beamw 		= beamw / px #Resize beam width from nm to pixels
		r 			= r / px #Resize radius from nm to pixel
		sumR 		= np.sum(img,1)/img.shape[1] 
		sumC 		= np.sum(img,0)/img.shape[0]
		sumRs 		= np.convolve(np.ones(SMOOTH),sumR,mode='same') / SMOOTH 
		sumCs 		= np.convolve(np.ones(SMOOTH),sumC,mode='same') / SMOOTH
		#Determine the beam orientation 
		maxR 		= np.argsort(sumRs)[-1::-1]
		maxC 		= np.argsort(sumCs)[-1::-1]
		
		if sumRs[maxR[0]] > sumCs[maxC[0]]:
			#Beam is horizontal
			isH 	= True
		else:
			#Rotate the image by 90 degrees using the transpose
			isH 	= False 
			maxR 	= maxC 
			sumR 	= sumC 
			sumRs 	= sumCs 
			img 		= img.transpose()
		
		#Go through sumRs and find local maxima with a local minima roughly halfway in between that is within +/- beamtol of beamw 
		if maxR[0] > maxR[1]:
			top 	= maxR[0] 
			bot 	= maxR[1] 
		else: 
			top 	= maxR[1] 
			bot 	= maxR[0] 	
		top0 	= top 
		bot0 	= bot
		i 		= 1
		while (np.abs(top - bot) < np.abs(beamw*(1-beamtol)) or np.abs(top-bot) > np.abs(beamw*(1+beamtol))) and i < maxR.shape[0]-1:
			i 	= i + 1 
			if maxR[i] > top:
				top  	= maxR[i] 
			elif maxR[i] < bot:
				bot 	= maxR[i]
			elif np.abs(top - maxR[i] - beamw) < np.abs(maxR[i] - bot - beamw):
				bot 	= maxR[i] 
			else:
				top 	= maxR[i]
		assert(i < maxR.shape[0] - 1)
	
		
		mxX 		= np.array([bot,top])
		mxY 		= np.array([sumR[bot],sumR[top]])
		lo 		= []
		hi 		= []
		x1 		= np.arange(0,img.shape[1]) 
		x2 		= x1 
			
		#Fit straight lines to the beam edge using a brute force method
		[p1,p2,p3] 	= self.__FindEdges__(img,bot,top,beamw,beamtol,DEBUG=False)
		
		if p2[1] - p1[1] < beamw*(1-beamtol):
			p1[1] 	= bot 
			p2[1] 	= top
		
		if DEBUG:
			plt.plot(px*np.arange(img.shape[0]),sumRs,'k-',lw=1,label='Sum R')
			#plt.plot(px*np.arange(img.shape[1]),sumC,'b-',lw=1,label='Sum C')
			plt.plot(px*mxX,mxY,'bs',ms=8)
			plt.plot(px*np.array([p1[1],p2[1]]),[sumR[p1[1]],sumR[p2[1]]],'ro',ms=8)
			plt.plot(px*np.array([p3[1]]),[sumR[p3[1]]],'yd',ms=8)
			plt.plot(px*np.array([top0,bot0]),sumR[[top0,bot0]],'mo',ms=8)
			plt.xlabel('Distance [nm]')
			plt.ylabel('Brightness [arb]')
			plt.legend(loc='upper right')
			plt.show()
			print "Beam width: %0.1f nm" % (px*(p2[1]-p1[1]))
			
		#Find the hole locations along the line 
		xf 		= np.arange(img.shape[1])
		yf 		= np.round(np.polyval(0.5*(p1+p2),xf),0).astype('int')
		I 		= img[yf,xf]
		Ismth 	= np.convolve(I, np.ones(SMOOTH),mode='same') / SMOOTH 
		#Find the local maxima that are spaced at least 2*r*(1-rtol) apart
		hC 		= self.__FindUnitCells__(xf[SMOOTH:-SMOOTH]-SMOOTH,Ismth[SMOOTH:-SMOOTH],r,rtol,DEBUG=DEBUG)
		hC 		= np.round((hC + SMOOTH),0).astype('uint16')
		if hC.shape[0] <= 1 :
			return [None, None, None, None, None, None]
		if DEBUG:
			plt.plot(px*xf,img[yf,xf],'k-',lw=1.5)
			plt.plot(px*xf,Ismth,'r-',lw=1.5)
			plt.plot(px*xf[hC],Ismth[hC],'bo',ms=8)
			plt.xlabel('Distance [nm]')
			plt.ylabel('Intensity [arb]')
			plt.show()

		#Fit the hole sizes
		rfit 		= []
		resid 	= []
		x0 		= []
		y0 		= []
		xCan 	= []
		yCan 	= []
		if EQHIST:
			imgEq 	= cv2.equalizeHist(img)
		else:
			imgEq 	= img
		for i in range(hC.shape[0]):
			if i == 0:
				xi1 		= 0
				xi2 		= 0.5*(hC[0] + hC[1])
			else:
				if i == hC.shape[0] - 1:
					xi2 	= img.shape[1]
				else:
					xi2 	= 0.5*(hC[i]+hC[i+1])
				xi1 		= 0.5*(hC[i-1] + hC[i])
			ylo 		= np.round(np.polyval(p1,np.arange(xi1,xi2)),0).astype('int')
			yhi 		= np.round(np.polyval(p2,np.arange(xi1,xi2)),0).astype('int')
			yi1 		= np.min(np.hstack([ylo,yhi]))
			yi2 		= np.max(np.hstack([ylo,yhi]))
			[x0t,y0t,rt,rsdt,xCt,yCt] 	= self.__FitCircle__(imgEq[(yi1+SMOOTH/2):(yi2-SMOOTH/2),(xi1):(xi2)],r,rtol,DEBUG=False)	
			if x0t is not None and np.abs(r-rt) <= r*rtol:
				x0.append(x0t+xi1)
				y0.append(y0t+yi1+SMOOTH/2)
				rfit.append(rt) 
				resid.append(100.*rsdt/rt) #Normalize the residual to a percent of the radius 
				xCan.append(xCt +xi1)
				yCan.append(yCt + yi1+SMOOTH/2)
		x0 		= np.array(x0)
		y0 		= np.array(y0)
		rfit 		= np.array(rfit)
		resid 	= np.array(resid)

		#Compute the average distance between the two beams
		m1 	= p1[0] 
		b1 	= p1[1]
		m2	= p2[0] 
		b2 	= p2[1] 
		x1 	= np.arange(img.shape[1])
		y1 	= np.polyval(p1,x1)
		x2 	= (x1 + m2*(m1*x1 + b1 - b2) ) / (1+m2**2)
		y2 	= m2*x2 + b2 
		dist 	= np.sqrt( np.power(x1-x2,2) + np.power(y1-y2,2) )
				
		color 	= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		for i in range(x0.shape[0]):
			color[yCan[i].astype('int'),xCan[i].astype('int')] 	= (0,255,0)
		plt.imshow(color)
		x1f 		= np.arange(img.shape[1])
		x2f 		= x1f 		
		plt.plot(x1f,np.polyval(p1,x1f),'m-',lw=2)
		plt.plot(x2f,np.polyval(p2,x2f),'m-',lw=2)
		plt.plot(x2f,np.polyval(0.5*(p1+p2),x2f),'b-',lw=2)
		for i in range(1,hC.shape[0]):
			xc 	= 0.5*(hC[i] - hC[i-1]) + hC[i-1]
			y1t 	= np.polyval(p1,xc)
			y2t 	= np.polyval(p2,xc)
			xi 	= xc*np.ones(20)
			yi 	= np.linspace(y1t,y2t,20)
			plt.plot(xi,yi,'y--',lw=1.5)
		th 	= np.linspace(0,2*math.pi,100)
		
		for i in range(x0.shape[0]):
			plt.plot(x0[i]+rfit[i]*np.cos(th),y0[i]+rfit[i]*np.sin(th),'r--',lw=1)
		#plt.xlabel('x [px]')
		#plt.ylabel('y [px]')
		plt.xlim([0,img.shape[1]])
		yvals 	= [np.min(np.polyval(p1,np.arange(img.shape[1]))),np.max(np.polyval(p2,np.arange(img.shape[1])))]
		plt.ylim(yvals)
		#Add the annotation
		ax 	= plt.gca()
		
		ax.annotate('200 nm', xy=(0.1*color.shape[1],0.75*yvals[1]+0.25*yvals[0]), xytext=(0.1*color.shape[1]+200,0.75*yvals[1]+0.25*yvals[0]-5), arrowprops=dict(facecolor='black', shrink=0.05,width=2,frac=0,headwidth=2))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)		
		if DEBUG:
			plt.show()					
			print "Distance is %d +- %d nm" % (px*np.mean(dist), px*np.std(dist))
		else:
			filen 	= "%s_fit.jpg" % ("".join(self.sem.GetFilename().split('.tif')))
			plt.savefig(filen)
			plt.close()
		return [px*x0,px*y0,px*rfit,px*resid,px*np.mean(dist),px*np.std(dist)]		
			
	def __FitCircle__(self,imgB,r,rtol,Nerode=2,DEBUG=False):
		#imgB 		= cv2.GaussianBlur(imgB,(5,5),0)
		ret3,th3 	= cv2.threshold(imgB,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		kernel 	= np.ones((Nerode,Nerode),np.uint8)
		opening 	= cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
		edges 	= cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, np.ones((2,2),np.uint8))
		edges 		= cv2.Canny(imgB,100,200)
		#Extract the approximate center of the circle
		[xg,yg] 	= np.meshgrid(np.arange(imgB.shape[1]),np.arange(imgB.shape[0]))
		xi 		= xg[edges>0].flatten()
		yi 		= yg[edges>0].flatten()
		if xi.shape[0] < 3:
			return [None,None,None,None,None,None]
		x0 		= np.mean(xi)
		y0 		= np.mean(yi)
		def FitFxn(a,xi,yi):
			x0 	= a[0] 
			y0 	= a[1]
			r 	= a[2]
			return np.sqrt(np.abs(np.power(xi-x0,2) + np.power(yi-y0,2) - r**2))
		f 		= spop.leastsq(lambda x: FitFxn(x,xi,yi),[x0,y0,r])
		if f[1]:
			atst 	= f[0]
			#Filter out all points not within +/- rtol*r of r 
			dvec 	= FitFxn(atst,xi,yi)
			xi2 		= xi[ dvec <= rtol*r]
			yi2 		= yi[ dvec <= rtol*r]
			if xi2.shape[0] < 3:
				x0 	= np.abs(f[0][0])
				y0 	= np.abs(f[0][1])
				r 	= np.abs(f[0][2])
				xi2 	 = xi 
				yi2 	 = yi
			else:
				f2 		= spop.leastsq(lambda x: FitFxn(x,xi2,yi2),atst)
				if f2[1]:
					x0 	= np.abs(f2[0][0])
					y0 	= np.abs(f2[0][1])
					r 	= np.abs(f2[0][2])
				else:
					x0 	= np.abs(f[0][0])
					y0 	= np.abs(f[0][1])
					r 	= np.abs(f[0][2])
					xi2 		= xi 
					yi2 		= yi
		else:
			if DEBUG:
				color 	= cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
				dedges 	= cv2.dilate(edges,kernel)
				color[dedges > 0] 	= (0,0,255)
				cv2.imshow('Fail',color)
				cv2.waitKey(0)
				cv2.destroyAllWindows()				
			return [None,None,None,None,None,None]
		
		if DEBUG:
			color 	= cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
			cv2.circle(color,(int(np.round(x0,0)), int(np.round(y0,0))), int(np.round(r,0)),(0,255,0),2)
			dedges 	= cv2.dilate(edges,kernel)
			color[dedges > 0] 	= (0,0,255)
			cv2.imshow('Circle Detection',color)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			#sys.exit()
		iord 		= np.argsort(np.arctan2(yi2,xi2))
		#The residual is given by the normalized standard deviation of the brightness around the spot edge 
		th 		= np.linspace(0,2*math.pi,1000)
		xi 		= np.round(x0 + r*np.cos(th),0).astype('uint16')
		yi 		= np.round(y0 + r*np.sin(th),0).astype('uint16')
		crd 		= 1e4*xi + yi 
		ucrd 		= np.unique(crd)
		yi 		= (ucrd % 1e4).astype('uint16')
		xi 		= np.floor(ucrd / 1e4).astype('uint16')
		vi 		= np.nonzero(np.logical_and( xi > 0, np.logical_and(yi > 0, np.logical_and(xi < imgB.shape[1], yi < imgB.shape[0]))))[0]
		
		if vi.shape[0] > 0:
			imgB 		= cv2.equalizeHist(imgB)
			edgeBright 	= imgB[yi[vi],xi[vi]]
		else:
			edgeBright 	= np.array([0])
		resid 		= np.std(edgeBright)/np.mean(edgeBright)
		#resid 		= FitFxn([x0,y0,r],xi2[iord],yi2[iord])
		return [x0,y0,r,resid,xi2[iord],yi2[iord]]
		
if __name__=="__main__":
	folder 		= "C:\\Users\\iroussea\\Documents\\2015-11-12 - A340x HSQ"
	img01 		= "img_00.tif"
	img03 		= "img_03.tif"
	img15 		= "img_15.tif"
	if len(sys.argv) == 3:
		imgID 		= int(sys.argv[1])
		r 			= float(sys.argv[2])
	else:
		imgID 		= 1 
		r 			= 45

	semi 		= ZeissSEMImage("%s\\img_%02d.tif" % (folder,imgID))
	nbh 		= NanobeamHoles(semi,r,0.1,200,0.2)
	dvec 		= nbh.GetR()
	rdvec 	= nbh.GetResid()
	print "Hole radius is %0.1f +/- %0.1f nm.  Resid is %0.2f +/- %0.2f" % (np.mean(dvec),np.std(dvec),np.mean(rdvec),np.std(rdvec))
	print "Beam width: %0.1f +- %0.1f" % (nbh.GetBeamWidth(),nbh.GetBeamWidthStd())
	
	