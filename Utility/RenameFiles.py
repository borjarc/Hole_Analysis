from os import listdir, rename, mkdir, chdir
from os.path import isfile, join, exists, splitext, isdir
import sys 
from shutil import move

def RenameAllFiles(dirname):
	chdir(dirname)
	onlyfile 	= [f for f in listdir(dirname) if isfile(join(dirname, f))]
	for fi in onlyfile:
		if len(fi.split(".")) == 1:
			newfilen 	= "%s.dat" % (fi)
			print "Moving %s to %s" % (fi, newfilen)
			move(fi, newfilen)
	chdir('..')

def SearchDirectory(dirname):
	d 		= dirname
	onlydir 	= [join(d,o) for o in listdir(d) if isdir(join(d,o))]
	for mydir in onlydir:
		SearchDirectory(mydir)
		RenameAllFiles(mydir) 
	RenameAllFiles('.')
	
if __name__=="__main__":
	dirname 	= "D:/2016-04-28/"
	SearchDirectory(dirname)
	
	