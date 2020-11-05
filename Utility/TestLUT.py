from UncertaintyLUT import *
import cPickle as pickle

if __name__=="__main__":
	ifile 		= "uLUT_sig1.pkl"
	lut 		= pickle.load(open(ifile,'rb'))
	lut.PlotFWHMstd()