#!/usr/bin/env python
import numpy as np
import idlsave
import math
import operator
import os
import sys 
import glob
import matplotlib.image as mpimg
import operator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from helperFn_2 import *
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

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
boundsiz=100   #if the size of the boundary points are less than <boundsiz> then eliminate the spoke
minrowsiz=10   #if the row size of the boundary points are less than <minrowsiz> then eliminate the spoke

##### parameters for expanding small spokes
expand_thread=2000 	# if less than <expand_thread> pixels, then expand
spoke_pix=0.02 		# if the brighness of the pixals next to that of the darkest spot of the spokes is within *spoke_pix* fraction then it is also a spoke pixal
pixelcheck=1 		# how many neighbor pixels to check

totchange_pix=1 	# how many to stop adding to the list
totchange_pixmax= 10 	# how many fraction of the total pixal increase is max increase and will break after that (this idicates the threadhold is too lose, result not converging)
iteration=5000 		# while adding pixels to spokes, if the iteration is greater than 100, break...





#filename='/Users/lucy/Desktop/spokes/SPKMVLFLP_081/W1597971520_1_cal.rpj1'
filename=sys.argv[1]

# read files and set up initial conditions
a=idlsave.read(filename)

datapoints=a.rrpi # load data in

if filename!='W1597976395_1_cal.rpj1':
	lon_array,rad_array,datapoints=cropdata(datapoints,a.mxlon,a.mnlon,a.mxrad,a.mnrad)
	m,n=datapoints.shape # get the shape of the brightness data
else:
	print('special file')
	m,n=datapoints.shape # get the shape of the brightness data
	lon_array=np.linspace(a.mnlon,a.mxlon,n)
	lon_array=lon_array[100:len(lon_array)]
	rad_array=np.linspace(a.mnrad,a.mxrad,m)
	datapoints=datapoints[100:len(lon_array),:]
	m,n=datapoints.shape # get the shape of the brightness data
	
print('image size: ',(m,n))
totpix=m*n

print('total number of pixcel:, ',totpix)

################## get image data/median and save in file  ####################################

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

# smooth background with median 
for i in range(m):
	med=np.median(datapoints[i,:])
	datapoints[i,:]=[(datapoints[i,j]-med) for j in range(n)]
	f.write(str(rad_array[i])+' '+str(med)+'\n')
f.close()

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

#plt.yscale('log')
#plt.show()
'''
plt.subplot(3,1,3)
datapoints=reduce_highFn(datapoints,cutoff=0.005)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.title('LPF')
datapoints=blur_image(datapoints, smooth_pix/2)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
datapointsbright=datapoints # copy data to identify the bright spots
plt.title('gaussian filter')
#datapoints=reduce_highFn(datapoints,cutoff=0.005)
#plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
#plt.title('high passed filter')
'''
'''
plt.subplot(4,1,4)
# smooth again (maybe use another method, reason of doing this: sometimes the peaks are too close to each other and the peak finding algorithm counts that as two peaks and this will identify some long dark spokes wrong. Could add somthing in the getrow funtion to say if two peaks are very close to each other then they are the same peak, make sure to also redifine the width and prominence/heights and stuff...)
datapoints=blur_image(datapoints, smooth_pix/2)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.title('gaussian filter 2')
'''

'''
#plt.show()
# find hot pixels...
plt.figure()
plt.subplot(3,1,1)
plt.imshow(datapointsbright, cmap = plt.get_cmap('gray'),origin='upper')
plt.subplot(3,1,2)
brightspokedad=np.zeros([m,n])
plt.imshow(datapointsbright, cmap = plt.get_cmap('gray'),origin='upper')
brightspokedad=getspokes_row2(-datapointsbright,brightspokedad,0.005,[10,150],1) # find peak in scipy
plt.subplot(3,1,3)
plt.imshow(datapointsbright, cmap = plt.get_cmap('gray'),origin='upper')
brighti,brightj=np.where(brightspokedad==spkcount)
plt.plot(brightj,brighti,'b.')
'''

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



print('getting rows')
# get row numbers with spokes
#datapoints=smoothdat(datapoints,smooth_pix)
spokesdata=np.zeros([m,n])
plt.figure()
plt.subplot(4,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.subplot(4,1,2)
#plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
spokeind=getdark(datapoints,spokesdata,getd)
plt.imshow(spokeind)
plt.title('getting darkest')
plt.subplot(4,1,3)
spokeind=findbound(spokeind)
spokeind=findspoke_num(spokeind,boundsiz,minrowsiz)
plt.imshow(spokeind)
maxs=int(max(spokeind.flatten()))-bound-1
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


print('finished')
#plt.show()
print('color different spokes in')
spokeind=getint_nr(spokeind)
print('finished coloring different spokes in')
spknum_range=range(bound*2+1,int(max(spokeind.flatten())+1))
print('spokes No.',len(spknum_range))
plt.subplot(4,1,4)
plt.imshow(spokeind,vmin=spknum_range[0]-1,vmax=spknum_range[-1]+1)

plt.savefig(filename+'darkest.png')


plt.show()


'''
# write results
print('writing results to file')
spkcount=0
filenamesafe=filename+'spokes.txt'
with open(filenamesafe,'w') as f:
	f.write('#rad lon intensity(subtracted median) spokes_number\n#note:spoke_number '+str(exp_spk)+' indicates pixels that are expanded\n')
	for i in spknum_range:
		spkcount=spkcount+1
		wherespk=np.where(spokeind==i)
		if len(wherespk[0])!=0:
			for j in range(len(wherespk[0])):
				#print(j)	
				f.write(str(rad_array[wherespk[0][j]])+' '+str(lon_array[wherespk[1][j]])+' '+str(datapoints[wherespk[0][j],wherespk[1][j]])+' '+str(spkcount)+'\n')

	whereexp=np.where(spokeind==exp_spk)
	if len(whereexp[0])!=0:
		for j in range(len(whereexp[0])):
			f.write(str(rad_array[whereexp[0][j]])+' '+str(lon_array[whereexp[1][j]])+' '+str(datapoints[whereexp[0][j],whereexp[1][j]])+' '+str(exp_spk)+'\n')


f.close()
print('finished')
'''


