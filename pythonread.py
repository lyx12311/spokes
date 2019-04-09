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

##### row threadholds
thmn=0.01 # min threadhold for fraction of pixals that are below median
thmx=1 # max threadhold for fraction of pixals that are below median
th_med=0.95 # threadhold to identify as posible spokes (th_med*median) will be count towards the pixals that are spokes
dk_ring_med=0.1 # if the median of this row is lower than this number, then it is just a dark ring not contain any spokes or it is hard to identify
spokes_sep_r=0.05 # if seperated more than *spokes_sep* fraction of the total observed longtitude then this is another spoke
shortspokes_r=0.05 # if the number of pixals in this row that contains spokes is smaller than *shortspokes_r* fraction of the total observed longtitude then it is probably just noise...


##### column threadholds
th_col=0.051 # how much deviation from mean when identify spokes columns
spokes_sep_c=0.01 # see *spokes_sep_r* for reference
shortspokes_c=0.01 # see *shortspokes_r* for reference

##### parameters for finding spokes after identifying the "spoke boxes"
stdvalue=0.002 # there is spokes in this row if the standard deviation is larger than 0.002 while calculating minimum to determine the darkest pixels

qufit=0.2  # for clustering while finding extra spokes (should be between [0, 1] 0.5 means that the median of all pairwise distances is used.)

spoke_pix=0.05 # if the brighness of the pixals next to that of the darkest spot of the spokes is within *spoke_pix* fraction then it is also a spoke pixal
pixelcheck=3 # how many neighbor pixels to check
totchange_pix=1 # how many to stop adding to the list
totchange_pixmax= 0.01 # how many fraction of the total pixal increase is max increase and will break after that (this idicates the threadhold is too lose, result not converging)
iteration=100 # while adding pixels to spokes, if the iteration is greater than 100, break...




# read files and set up initial conditions
a=idlsave.read('W1597976395_1_cal.rpj1')

datapoints=a.rrpi # load data in
m,n=datapoints.shape # get the shape of the brightness data
print('image size: ',(m,n))
totpix=m*n
print('total number of pixcel:, ',totpix)
m,n=datapoints.shape # get the shape of the brightness data

# get row numbers with spokes
withspi=getrows(datapoints,th_med,dk_ring_med,thmn,thmx,spokes_sep_r,shortspokes_r)
withspi_1d=np.hstack(withspi) # put into 1d

# get column numbers with spokes
withspj=getcols(withspi,datapoints,th_col,spokes_sep_c,shortspokes_c)
withspj_1d=np.hstack(withspj) # put into 1d

if len(withspi_1d)==0 or len(withspj_1d)==0:
	print("No spokes!")
	exit(1)

################## plotting original image and smoothed image ####################################
# 2D original plot without spokes 
plt.figure()
plt.title('original image')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'))

# smooth background with median 
plt.figure()
plt.title('original image - median of each row')
for i in range(m):
	med=np.median(datapoints[i,:])
	datapoints[i,:]=datapoints[i,:]-med
plt.imshow(datapoints, cmap = plt.get_cmap('gray'))
############################################################################################################


# cropped the data so only ones with spokes are left
datacrop=datapoints[np.reshape(withspi_1d,(len(withspi_1d),1)),np.reshape(withspj_1d,(1,len(withspj_1d)))]

# spoke boxes
spkbox_prebpx=[[] for i in range(len(withspi)*len(withspj))] # pixal values
spkbox_ind_lon_prebpx=[[] for i in range(len(withspi)*len(withspj))] # pixal indices in lon
spkbox_ind_rad_prebpx=[[] for i in range(len(withspi)*len(withspj))] # pixal indices in rad
numb=0
for i in range(len(withspi)):
	for j in range(len(withspj)):
		spkbox_prebpx[numb]=datapoints[min(withspi[i]):max(withspi[i]),min(withspj[j]):max(withspj[j])]
		spkbox_ind_lon_prebpx[numb]=range(min(withspj[j]),max(withspj[j]))
		spkbox_ind_rad_prebpx[numb]=range(min(withspi[i]),max(withspi[i]))
		numb=numb+1

################## subplot of all the data ################
# plot subplots of all the data
'''
# find overall min
plt.figure()
spokes_ind_lon=[] # get spokes center in lon
spokes_ind_rad=[] # get spokes center in rad
minidx=[]
subpltn=1 # subplot counts
for i in range(len(withspi)*len(withspj)):
	plt.subplot(len(withspi),len(withspj),subpltn)
	plt.imshow(spkbox[i], cmap = plt.get_cmap('gray'))
	locallon=[]
	spkbox[i]=np.array(spkbox[i])
	spkflat=spkbox[i].flatten()
	#minidx=(np.argsort(spkflat)[:spkpeak])
	minidx=(np.argsort(spkflat))
	k=0
	unrmin=np.unravel_index(minidx[k], spkbox[i].shape)
	longind=unrmin[1]
	locallon.append(longind)
	spokes_ind_lon.append(spkbox_ind_lon[i][unrmin[1]])
	spokes_ind_rad.append(spkbox_ind_rad[i][unrmin[0]])
	k=1
	indcout=1
	while k<spkpeak and indcout<len(minidx):
		unrmin=np.unravel_index(minidx[k], spkbox[i].shape)
		longind=unrmin[1]
		if longind not in locallon:
			spokes_ind_lon.append(spkbox_ind_lon[i][unrmin[1]])
			spokes_ind_rad.append(spkbox_ind_rad[i][unrmin[0]])
			locallon.append(longind)
			plt.plot(unrmin[1],unrmin[0],'bo')
			k=k+1
			indcout=indcout+1
		else:
			indcout=indcout+1
	subpltn=subpltn+1
'''
'''
# find min in each row and column
plt.figure()
spokes_ind_lon=[] # get spokes center in lon
spokes_ind_rad=[] # get spokes center in rad
minidx=[]
subpltn=1 # subplot counts
for i in range(len(withspi)*len(withspj)):
	plt.subplot(len(withspi),len(withspj),subpltn)
	plt.imshow(spkbox[i], cmap = plt.get_cmap('gray'))
	locallon=[]
	spkbox[i]=np.array(spkbox[i])
	for k in range(len(spkbox[i])):
		index, value = min(enumerate(spkbox[i][k]), key=operator.itemgetter(1))
		spokes_ind_lon.append(spkbox_ind_lon[i][index])
		spokes_ind_rad.append(spkbox_ind_rad[i][k])
		plt.plot(index,k,'bo')
	for j in range(len(spkbox[i][:][0])):
		rownum=[spkbox[i][p][j] for p in range(len(spkbox[i]))]
		index, value = min(enumerate(rownum), key=operator.itemgetter(1))
		spokes_ind_lon.append(spkbox_ind_lon[i][j])
		spokes_ind_rad.append(spkbox_ind_rad[i][index])
		plt.plot(j,index,'ro')
	subpltn=subpltn+1
'''

# find min in each row
spokesnumb=len(withspi)*len(withspj) # spokes number
plt.figure()
plt.title('spoke boxes and spokes before clustering')
spokes_ind_lon_prebpx=[[] for i in range(spokesnumb)] # get spokes center in lon for each spoke
spokes_ind_rad_prebpx=[[] for i in range(spokesnumb)] # get spokes center in rad for each spoke
minidx=[]
subpltn=1 # subplot counts
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, spokesnumb))
for i in range(spokesnumb):
	plt.subplot(len(withspi),len(withspj),subpltn)
	plt.imshow(spkbox_prebpx[i], cmap = plt.get_cmap('gray'))
	locallon=[]
	spkbox_prebpx[i]=np.array(spkbox_prebpx[i])
	for k in range(len(spkbox_prebpx[i])):
		index, value = min(enumerate(spkbox_prebpx[i][k]), key=operator.itemgetter(1))
		if np.std(spkbox_prebpx[i][k])<stdvalue:
			continue
		else:
			spokes_ind_lon_prebpx[i].append(spkbox_ind_lon_prebpx[i][index])
			spokes_ind_rad_prebpx[i].append(spkbox_ind_rad_prebpx[i][k])
			plt.plot(index,k,'o',c=colors[i])
	subpltn=subpltn+1

# check if there are other spokes within one box (clustering)
plt.figure()
plt.title('spokes after clustering')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'))
distant=[[] for i in range(spokesnumb)]
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

spokes_ind_rad_1d=np.hstack(spokes_ind_rad_prebpx)
spokes_ind_lon_1d=np.hstack(spokes_ind_lon_prebpx)

centers = [spokes_ind_rad_prebpx[0][0], spokes_ind_rad_prebpx[0][0]]
X=np.array(zip(spokes_ind_rad_1d,spokes_ind_lon_1d))
bandwidth = estimate_bandwidth(X,quantile=qufit)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
spkbox=[[] for i in range(n_clusters_)] # pixal values
spokes_ind_lon=[[] for i in range(n_clusters_)] # get spokes center in lon for each spoke
spokes_ind_rad=[[] for i in range(n_clusters_)] # get spokes center in rad for each spoke
for k, col in zip(range(n_clusters_), colors):
	my_members = labels == k
	# reasign boxes
	spokes_ind_lon[k]=X[my_members, 1]
	spokes_ind_rad[k]=X[my_members, 0]
	spkbox[k]=datapoints[min(spokes_ind_rad[k]):max(spokes_ind_rad[k]),min(spokes_ind_lon[k]):max(spokes_ind_lon[k])]
	
    	cluster_center = cluster_centers[k]
    	plt.plot(X[my_members, 1], X[my_members, 0], 'o')
    	plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)

spokesnumb=n_clusters_

print('length of lon: ',len(spokes_ind_lon[0]))

# get the spokes (check if the surrounding pixals are part of the spokes)
countnew=0

# average increase in pixels
totchange=[totchange_pix+1 for j in range(spokesnumb)]
totchangep=[totchange_pix+1 for j in range(spokesnumb)]

# each direction count
totchange_ev_p=[0 for i in range(6)]

britness=[0 for j in range(spokesnumb)]
spokes_ind_lon_newadd=[[] for j in range(spokesnumb)]
spokes_ind_rad_newadd=[[] for j in range(spokesnumb)]

while countnew<iteration and any([totchange[i]>totchange_pix for i in range(spokesnumb)]) and any([totchange[i]<totpix*totchange_pixmax for i in range(spokesnumb)]): #and any(diff[i]>totchange_pix for i in range(6)):
	
	totchange=[0 for j in range(spokesnumb)]
	totchange_ev=[0 for j in range(6)]
	for k in range(spokesnumb):
		if totchangep[k]>totchange_pix and totchangep[k]<totpix*totchange_pixmax:
			if countnew==0:
				spokes_ind_lon_newadd_p=spokes_ind_lon[k]
				spokes_ind_rad_newadd_p=spokes_ind_rad[k]
				#britness[k]=np.mean([datapoints[a][b] for a,b in zip(spokes_ind_rad_newadd_p,spokes_ind_lon_newadd_p)])  # get average brightness of the spoke
			else:
				spokes_ind_lon_newadd_p=spokes_ind_lon_newadd[k]
				spokes_ind_rad_newadd_p=spokes_ind_rad_newadd[k]
				spokes_ind_lon_newadd[k]=[]
				spokes_ind_rad_newadd[k]=[]
			#print('brightness is:',britness)
			for i in range(len(spokes_ind_lon_newadd_p)):
				x=spokes_ind_lon_newadd_p[i]
				y=spokes_ind_rad_newadd_p[i]
				britness[k]=datapoints[y][x]
				for np1 in range(pixelcheck):
					if (y+np1+1<m) and ((x,y+np1+1) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y+np1+1][x]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y+np1+1)
						spokes_ind_lon_newadd[k].append(x)
						spokes_ind_rad_newadd[k].append(y+np1+1)
						totchange[k]=totchange[k]+1
						totchange_ev[0]=totchange_ev[0]+1
			
					if (x+np1+1<n) and ((x+np1+1,y) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y][x+np1+1]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x+np1+1)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y)
						spokes_ind_lon_newadd[k].append(x+np1+1)
						spokes_ind_rad_newadd[k].append(y)
						totchange[k]=totchange[k]+1
						totchange_ev[1]=totchange_ev[1]+1
					
					if (x+np1+1<n and y+np1+1<m) and ((x+np1+1,y+np1+1) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y+np1+1][x+np1+1]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x+np1+1)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y+np1+1)
						spokes_ind_lon_newadd[k].append(x+np1+1)
						spokes_ind_rad_newadd[k].append(y+np1+1)
						totchange[k]=totchange[k]+1
						totchange_ev[2]=totchange_ev[2]+1
			
					if (y-np1-1>0) and ((x,y-np1-1) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y-np1-1][x]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y-np1-1)
						spokes_ind_lon_newadd[k].append(x)
						spokes_ind_rad_newadd[k].append(y-np1-1)
						totchange[k]=totchange[k]+1
						totchange_ev[3]=totchange_ev[3]+1
				
					if (x-np1-1>0) and ((x-np1-1,y) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y][x-np1-1]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x-np1-1)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y)
						spokes_ind_lon_newadd[k].append(x-np1-1)
						spokes_ind_rad_newadd[k].append(y)
						totchange[k]=totchange[k]+1
						totchange_ev[4]=totchange_ev[4]+1
				
					if (x-np1-1>0 and y-np1-1>0) and ((x-np1-1,y-np1-1) not in zip(spokes_ind_lon[k],spokes_ind_rad[k])) and (abs(datapoints[y-np1-1][x-np1-1]-britness[k])<abs(spoke_pix*britness[k])):
						spokes_ind_lon[k]=np.append(spokes_ind_lon[k],x-np1-1)
						spokes_ind_rad[k]=np.append(spokes_ind_rad[k],y-np1-1)
						spokes_ind_lon_newadd[k].append(x-np1-1)
						spokes_ind_rad_newadd[k].append(y-np1-1)
						totchange[k]=totchange[k]+1
						totchange_ev[5]=totchange_ev[5]+1
	totchange_ev_p=totchange_ev
	print('Total pixal changes per spoke: ',totchange)
	print('Total pixal changes per direction: ',totchange_ev)
	totchangep=totchange
	countnew=countnew+1
	
if any([totchange[i]>totpix*totchange_pixmax for i in range(spokesnumb)]):
	print('Error: Choose a smaller spoke_pix value so result can converage!')
	print('Total change: ',totchange)
	exit(1)		
elif countnew>=iteration:
	print("Warning: Result hasn't converage after ",iteration," iteration")
else:
	print("Finished identifying spokes")
	
# 2D original plot with spokes 
plt.figure()
colors = cmap(np.linspace(0, 1, spokesnumb))
plt.title('original image with identified spokes')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'))
print('length of lon: ',len(spokes_ind_lon[0]))
for i in range(spokesnumb):
	plt.plot(spokes_ind_lon[i],spokes_ind_rad[i],'o',c=colors[i])

# plot lines to show which columns and rows are identified as having spokes
plt.figure()
plt.imshow(datapoints, cmap = plt.get_cmap('gray'))
for i in withspi_1d:
	plt.plot(np.arange(n),i*np.ones(n),'b')
for j in withspj_1d:
	plt.plot(j*np.ones(m),np.arange(m),'r')

plt.show()


