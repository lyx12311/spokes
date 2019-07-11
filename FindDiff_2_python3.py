#!/usr/local/bin/python3
import numpy as np
import idlsave
import math
import operator
import os
import sys 
import glob
#import matplotlib.image as mpimg
import operator
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from helperFn_2_python3 import *
from itertools import cycle
# only for clustering (not used now)
from sklearn.cluster import MeanShift, estimate_bandwidth 
from sklearn.datasets.samples_generator import make_blobs 


spkcount=1
sharpedge=-2
bound=100
# assigned bound ID number: 100+
nonspk=-1000
# assigned spokes ID number: 200+
exp_spk=999 # after expanding pixel numbers
peaks_ind=500

'''
code---
non-spokes: -1
spokes:1-50
not processed: 0
sharp edge: -2
boundaries without identified as spokes: 100
'''
##### spokes threadholds
smooth_pix= 50		# how many pixel to smooth
passfiltrow=0		# how many rows to get rid of in 2d fft
passfiltcol=3		# how many cols to get rid of in 2d fft
boundsiz=100   #if the size of the boundary points are less than <boundsiz> then eliminate the spoke
minrowsiz=10   #if the row size of the boundary points are less than <minrowsiz> then eliminate the spoke


##### parameters for expanding small spokes
expand_thread=2000 	# if less than <expand_thread> pixels, then expand
spoke_pix=0.02 		# if the brighness of the pixals next to that of the darkest spot of the spokes is within *spoke_pix* fraction then it is also a spoke pixal
pixelcheck=1 		# how many neighbor pixels to check

totchange_pix=1 	# how many to stop adding to the list
totchange_pixmax= 10 	# how many fraction of the total pixal increase is max increase and will break after that (this idicates the threadhold is too lose, result not converging)
iteration=5000 		# while adding pixels to spokes, if the iteration is greater than 100, break...



# check command line arguments 
def checkinput(argv):                                                                       
	programname = sys.argv[0]                                                               
	if len(argv) != 2:  # Exit if not exactly one arguments  
		print('---------------------------------------------------------------------------')                               
		print('This program is a pipeline for finding spokes after the data has been alined and stored into idl .rpj1 files.\n It uses 2d fft to reduce the data, it then finds the darkest pixels and consider as a spokes pixel')
		print('It takes into the file name as argument.\n Output: one plot with the data reduction process and result and one file that gives the numerical information of the processed data')
		print(' ')
		print(' Example:    '+programname+' SPKMVLFLP_081/W1597987120_1_cal.rpj1') 
		print('---------------------------------------------------------------------------')                                    
		sys.exit(1)                                                                         
	gridfile = argv[1]                                                                                                                                    
	if not os.path.isfile(gridfile):  # Exit if folder does not exist                  
		print('ERROR: unable to locate file ' + gridfile)                             
		sys.exit(1)                                                                            
checkinput(sys.argv)




#filename='/Users/lucy/Desktop/spokes/SPKMVLFLP_081/W1597971520_1_cal.rpj1'
filename=sys.argv[1]

# read files and set up initial conditions
a=idlsave.read(filename)

datapoints=a.rrpi # load data in

if filename!='W1597976395_1_cal.rpj1':
	lon_array,rad_array,datapoints=cropdata(datapoints,a.mxlon,a.mnlon,a.mxrad,a.mnrad)
	m_i,n_i=datapoints.shape # get the shape of the brightness data
	m,n=datapoints.shape # get the shape of the brightness data
else:
	print('special file')
	m,n=datapoints.shape # get the shape of the brightness data
	lon_array=np.linspace(a.mnlon,a.mxlon,n)
	lon_array=lon_array[100:len(lon_array)]
	rad_array=np.linspace(a.mnrad,a.mxrad,m)
	datapoints=datapoints[100:len(lon_array),:]
	m_i,n_i=datapoints.shape # get the shape of the brightness data
	m,n=datapoints.shape # get the shape of the brightness data
	
print(('image size: ',(m,n)))
totpix=m*n

print(('total number of pixcel:, ',totpix))

################## get image data/median and save in file  ####################################
import copy 
dataor=copy.copy(datapoints) # avoid deep copy issue... not sure why its happending
'''
# print out median, don't need this 
filenameimg=filename+'image.txt'
f=open(filenameimg,'w')
f.write('#image size: '+'('+str(m)+','+str(n)+')'+'\n')
f.write('#total number of pixcel:, '+str(totpix)+'\n')
f.write('#minLon maxLon minRad maxRad\n')
f.write(str(a.mnlon)+' '+str(a.mxlon)+' '+str(a.mnrad)+' '+str(a.mxrad)+'\n')
f.write('########################################### below are median intensity for each rad #################################################################\n')
f.write('#Rad Median\n')
plt.figure()
plt.imshow(datapoints,cmap = plt.get_cmap('gray'),origin='upper')
#plt.show()
'''
# smooth background with median 
for i in range(m):
	med=np.median(datapoints[i,:])
	datapoints[i,:]=[(datapoints[i,j]-med) for j in range(n)]
	#f.write(str(rad_array[i])+' '+str(med)+'\n')
#f.close()

minda=(min(datapoints.flatten()))
for i in range(m):
	datapoints[i,:]=[(datapoints[i,j]+abs(minda)) for j in range(n)]



############################################################################################################


##### test differnt smoothing.....######
############################################################################################################
import cv2
# bluring image
plt.figure()
plt.subplot(2,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.subplot(2,1,2)
datapoints=fft2lpf(datapoints,passfiltrow,passfiltcol)
#datapoints[:,1600:n-1]=fft2lpf(datapoints[:,1600:n-1],passfiltrow,1)
datapoints=blur_image(datapoints, smooth_pix)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.title('2d fft')
#plt.show()

import matplotlib.mlab as mlab
# fit histogram to brightness so we can get the threadhold for getdark() function
plt.figure()
dataflat=datapoints.flatten()
plt.hist(dataflat,bins=np.linspace(min(dataflat),max(dataflat),200), normed=True)
#print(counts)
mean = np.mean(dataflat)
variance = np.var(dataflat)
sigma = np.sqrt(variance)
x = np.linspace(min(dataflat), max(dataflat), 100)
plt.plot(x, mlab.normpdf(x, mean, sigma))
getd=mean-sigma

m,n=datapoints.shape
lon_array=np.linspace(min(lon_array),max(lon_array),n)
rad_array=np.linspace(min(rad_array),max(rad_array),m)
'''
m_2,n_2=datapoints_2.shape
for i in range(m):
	if i==300:
		plt.figure()
		plt.subplot(2,1,1)
		plt.plot(datapoints[i,:])
		plt.title('with gaussian bluring')
		plt.subplot(2,1,2)
		plt.plot(datapoints_2[i,:])
		plt.title('with average box bluring')
		plt.show()
'''



# get row numbers with spokes
#datapoints=smoothdat(datapoints,smooth_pix)
spokesdata=np.zeros([m,n])
f,(ax1,ax2,ax3,ax4)=plt.subplots(4, sharex=True, sharey=True)
#print(dataor.shape)
#print(datapoints.shape)
ax1.imshow(dataor[0:datapoints.shape[0],0:datapoints.shape[1]], cmap = plt.get_cmap('gray'),origin='upper')

ax2.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
print('getting darkest')

spokeind=getdark(datapoints,spokesdata,getd)
ax3.imshow(spokeind)

spokeind=findbound(spokeind)
'''
plt.figure()
plt.imshow(spokeind)
plt.show()
'''
spokeind=findspoke_num(spokeind,boundsiz,minrowsiz)
'''
plt.imshow(spokeind)
maxs=int(max(spokeind.flatten()))-bound
#print(maxs)
jar=[[] for i in range(maxs)]
iar=[[] for i in range(maxs)]

#plt.figure()
#plt.imshow(spokeind)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for sp,col in zip(range(maxs), colors):
	for i in range(m):
		for j in range(n):
			if spokeind[i,j]==bound+1+sp:
				jar[sp].append(j)
				iar[sp].append(i)	
	plt.plot(jar[sp],iar[sp],'.',color=col)	
	
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()
'''

print('finished')
#plt.show()
print('coloring spokes in')
spokeind=getint_nr(spokeind)
print('finished coloring different spokes in')
spknum_range=list(range(bound*2+1,int(max(spokeind.flatten())+1)))
spknum_range,spokesdata=cleanedge_spk(spokesdata,spknum_range,0.1)
#print(spknum_range)
print(('spokes No.',len(spknum_range)))
f.subplots_adjust(hspace=0.1)
ax4.imshow(spokeind,vmin=spknum_range[0]-1,vmax=spknum_range[-1]+1)

plt.savefig(filename+'darkest.png')


#plt.show()



# write results
print('writing results to file')
spkcount=0
datamed=np.median(datapoints)
#print(datamed)
filenamesafe=filename+'spokes.txt'
with open(filenamesafe,'w') as f:
	f.write('#image data')
	f.write('#image size: '+'('+str(m_i)+','+str(n_i)+')'+'\n')
	f.write('#total number of pixcel:, '+str(totpix)+'\n')
	f.write('#minLon maxLon minRad maxRad\n')
	f.write(str(a.mnlon)+' '+str(a.mxlon)+' '+str(a.mnrad)+' '+str(a.mxrad)+'\n')
	f.write('# spokes data')
	f.write('#rad lon intensity(2d fft-image median) spokes_number\n#note:spoke_number '+str(exp_spk)+' indicates pixels that are expanded\n')
	for i in spknum_range:
		spkcount=spkcount+1
		wherespk=np.where(spokeind==i)
		if len(wherespk[0])!=0:
			for j in range(len(wherespk[0])):
				#print(j)	
				f.write(str(rad_array[wherespk[0][j]])+' '+str(lon_array[wherespk[1][j]])+' '+str(datapoints[wherespk[0][j],wherespk[1][j]]-datamed)+' '+str(spkcount)+'\n')
f.close()
#print('finished')



