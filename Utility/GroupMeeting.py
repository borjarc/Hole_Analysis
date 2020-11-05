import numpy.random as rand 
import numpy as np

def PickName(lst):
	ind 	= int(np.floor(rand.random(1)*float(len(lst))))
	print lst[ind]
	del lst[ind]
	return lst 

if __name__=="__main__":
	mems 	= ["Gordon","Kanako","Camille","Sebastian","Pirouz","Joachim","Ian","Hezhi","Antoine","Gwen","Wei"]
	for i in range(len(mems)):
		mems 	= PickName(mems)