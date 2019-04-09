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

##### how many fraction of darkest/brightess pixel to get rid of
db_per=0.01

sumup=5
threaddist=0.5e4

##### ratio
ctr_r=0.3

##### parameters for finding spokes after identifying the "spoke boxes"
stdvalue=0.002 # there is spokes in this row if the standard deviation is larger than 0.002 while calculating minimum to determine the darkest pixels

qufit=0.2  # for clustering while finding extra spokes (should be between [0, 1] 0.5 means that the median of all pairwise distances is used.)

spoke_pix=0.3 # if the brighness of the pixals next to that of the darkest spot of the spokes is within *spoke_pix* fraction then it is also a spoke pixal
pixelcheck=10 # how many neighbor pixels to check

totchange_pix=1 # how many to stop adding to the list
totchange_pixmax= 10 # how many fraction of the total pixal increase is max increase and will break after that (this idicates the threadhold is too lose, result not converging)
iteration=5 # while adding pixels to spokes, if the iteration is greater than 100, break...


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

# get darkest pixals
plt.figure()
plt.title('original image - median of each row')
for i in range(m):
	med=np.median(datapoints[i,:])
	datapoints[i,:]=datapoints[i,:]-med
plt.subplot(2,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')

plt.subplot(2,1,2)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
data_1d=sorted(datapoints.flatten())
data_count=data_1d[int(db_per*totpix):int((1-db_per)*totpix)]
#print(data_count)
lon_array_d=[]
rad_array_d=[]
for k in range(1000):
	minar=np.where(datapoints == (data_count[k]))
	lon_array_d.append(lon_array[int(minar[1][0])])
	rad_array_d.append(rad_array[int(minar[0][0])])
	plt.plot(minar[1][0],minar[0][0],'bo')

lon_array_dd=[]
rad_array_dd=[]
dist_ana=[]

for i in range(len(lon_array_d)):
	dist=sorted([(lon_array_d[i]-lon_array_d[j])**2.+(rad_array_d[i]-rad_array_d[j])**2. for j in range(len(lon_array_d))])
	dist_ana.append(sum(dist[0:sumup]))
	if sum(dist[0:sumup])<threaddist:
		lon_array_dd.append(np.where(lon_array==lon_array_d[i])[0])
		rad_array_dd.append(np.where(rad_array==rad_array_d[i])[0])
plt.plot(lon_array_dd,rad_array_dd,'ro')

plt.figure()
plt.hist(dist_ana,1000)
plt.yscale("log")
plt.show()


