from NanobeamHoles import * 
from SEMImage import *
import cPickle as pickle
import copy
import os.path

def GetHoleRadius(i):
	#Return the hole diameter for image <i> .tif
	if i >= 4 and i <= 13:
		return 50
	elif i >= 14 and i <= 21:
		return 45
	elif i > 21 and i < 33:
		return 40 
	elif i > 32 and i < 41:
		return 35
	elif i > 40 and i < 52:
		return 30
	elif i > 51 and i < 63:
		return 50
	elif i > 62 and i < 74:
		return 45 
	elif i > 73 and i < 85:
		return 40
	elif i > 84 and i < 94:
		return 35 
	else:
		return 30
		
def GetBeamWidth(i):
	#Return the beam width 
	if i >= 4 and i < 52:
		return 200
	else:
		return 230

if __name__=="__main__":
	folder 		= "U:/Fabrication/SEM Images/Merlin/2016-03-23-A3406 Etch Test"

	mydat 		= []
	nums 		= range(103)
	nums 		= filter(lambda x: not x == 22, nums)
	for i in nums:
		imgf 		= "%s/img_%02d.tif" % (folder,int(i))
		if not (os.path.exists(imgf)):
			print "%s not found" % (imgf)
			continue
		semi 		= ZeissSEMImage(imgf)
		#print "%s: %0.2f" % (imgf, semi.GetPixelSize())
		#semi.ShowImageCropped()
		#sys.exit()
		nbh 		= NanobeamHoles(semi,GetHoleRadius(i)-11,0.2,GetBeamWidth(i),0.2,70)
		r 			= nbh.GetR()
		print "IMG: %02d D: %0.1f +/- %0.1f nm" % (i, np.mean(r), np.std(r))
		mydat.append(nbh)		
		
	#pickle.dump(mydat,open('A3406-EtchDecember-2.pkl','wb'))
