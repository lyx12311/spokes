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
from helperFn import *
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

'''
code---
non-spokes: -1
spokes:1-50
not processed: 0
sharp edge: -2
boundaries without identified as spokes: 100
'''
##### spokes threadholds
smooth_pix= 10	# how many pixel to smooth
checkNN= 5	# check if it is also dark in this many neighboring pixels
thread=0.85 	# only check pixels that have brightness values lower than thread*median 

##### get boundary threadholds
shortsk=200
crid_r=10

##### parameters for expanding small spokes
expand_thread=1000 # if less than <expand_thread> pixels, then expand
spoke_pix=0.2 # if the brighness of the pixals next to that of the darkest spot of the spokes is within *spoke_pix* fraction then it is also a spoke pixal
pixelcheck=5 # how many neighbor pixels to check

totchange_pix=1 # how many to stop adding to the list
totchange_pixmax= 10 # how many fraction of the total pixal increase is max increase and will break after that (this idicates the threadhold is too lose, result not converging)
iteration=100 # while adding pixels to spokes, if the iteration is greater than 100, break...





#filename='/Users/lucy/Desktop/spokes/SPKMVLFLP_081/W1597971520_1_cal.rpj1'
filename=sys.argv[1]

# read files and set up initial conditions
a=idlsave.read(filename)

datapoints=a.rrpi # load data in

if filename!='W1597976395_1_cal.rpj1':
	lon_array,rad_array,datapoints=cropdata(datapoints,a.mxlon,a.mnlon,a.mxrad,a.mnrad)

m,n=datapoints.shape # get the shape of the brightness data
print('image size: ',(m,n))
totpix=m*n

print('total number of pixcel:, ',totpix)

spokesdata=np.zeros([m,n])

################## get image data and save in file  ####################################

filenameimg=filename+'image.txt'
f=open(filenameimg,'w')
f.write('#image size: '+'('+str(m)+','+str(n)+')'+'\n')
f.write('#total number of pixcel:, '+str(totpix)+'\n')
f.write('#minLon maxLon minRad maxRad\n')
f.write(str(a.mnlon)+' '+str(a.mxlon)+' '+str(a.mnrad)+' '+str(a.mxrad)+'\n')
f.write('########################################### below are median intensity for each rad #################################################################\n')
f.write('#Rad Median\n')

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
plt.figure()
plt.subplot(2,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.subplot(2,1,2)
plt.imshow(blur_image(datapoints, smooth_pix), cmap = plt.get_cmap('gray'),origin='upper')
plt.title('smoothed')
#plt.show()
#datapoints_2=smoothdat(datapoints,smooth_pix)
datapoints=blur_image(datapoints, smooth_pix)
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
spokeind=getspokes_row(datapoints,spokesdata,checkNN,thread)
plt.savefig(filename+'original.png')
############################################################################################################

maxsp=int(max(spokeind.flatten()))
#print(maxsp)
for i in range(maxsp):
	minar=np.where(spokeind != 0)
	spokes_ind_lon_1d=[int(minar[1][j]) for j in range(len(minar[1]))]
	spokes_ind_rad_1d=[int(minar[0][j]) for j in range(len(minar[0]))]

plt.figure()
plt.imshow(spokeind)
print('finding boundaries')
spokeind=findbound(spokeind)
print('finding different spoke boundaries')
spokeind=findspoke_num(spokeind,shortsk,crid_r)
maxs=int(max(spokeind.flatten()))-bound-1
#print(maxs)
jar=[[] for i in range(maxs)]
iar=[[] for i in range(maxs)]

plt.figure()
plt.imshow(spokeind)
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
plt.savefig(filename+'outline.png')

#getint(spokeind)
print('color different spokes in')
spokeind=getint_nr(spokeind)
print('finished coloring different spokes in')

spknum_range=range(bound*2+1,int(max(spokeind.flatten())+1))
print('spokes No.',len(spknum_range))
plt.figure()
plt.subplot(3,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.subplot(3,1,2)
plt.imshow(spokeind,vmin=spknum_range[0]-1,vmax=spknum_range[-1]+1)
plt.title('before expanding')
#plt.show()			
plt.title('without sharp edges')

print('expanding small spokes...')
# get absolute darkest spokes brightness 
brightness=[]
for i in range(len(spknum_range)):
	#print(spknum_range[i])
	if len(np.where(spokeind==spknum_range[i])[0])>0:
		#print(len(np.where(spokeind==spknum_range[i])[0]))
		brightness.append(np.mean(datapoints[np.where(spokeind==spknum_range[i])]))
#print(brightness)
brightness_m=min(brightness)
#print(brightness_m)
# expand on small spokes
for i in spknum_range:
	if len(np.where(spokeind==i))!=0 and len(np.where(spokeind==i))<expand_thread:
		print('spoke '+str(i)+' is expanding')
		spokeind=expand_spokes(datapoints,spokeind,i,iteration,brightness_m,spoke_pix,pixelcheck)
print('finished expanding small spokes...')

plt.subplot(3,1,3)
#spokeind=sortSpk(spokeind,spknum_range)
plt.imshow(spokeind,vmin=spknum_range[0]-1,vmax=spknum_range[-1]+1)
plt.title('after expanding')
plt.savefig(filename+'pre_fin.png')
#plt.show()



# write results
print('writing results to file')
# save into file
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
			#print(rad_array[whereexp[0][j]])
			#print(lon_array[whereexp[1][j]])
			#print(datapoints[whereexp[0][j],whereexp[1][j]])
			f.write(str(rad_array[whereexp[0][j]])+' '+str(lon_array[whereexp[1][j]])+' '+str(datapoints[whereexp[0][j],whereexp[1][j]])+' '+str(exp_spk)+'\n')


f.close()
print('finished')


#plt.show()


