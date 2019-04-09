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
thmn=0.005 # 0.005 default min threadhold for fraction of pixals that are below median
thmx=1 # max threadhold for fraction of pixals that are below median
th_med=0.955 # threadhold to identify as posible spokes (th_med*median) will be count towards the pixals that are spokes
dk_ring_med=0.01 # if the median of this row is lower than this number, then it is just a dark ring not contain any spokes or it is hard to identify
spokes_sep_r=0.05 # if seperated more than *spokes_sep* fraction of the total observed longtitude then this is another spoke
shortspokes_r=0.01 # if the number of pixals in this row that contains spokes is smaller than *shortspokes_r* fraction of the total observed longtitude then it is probably just noise...

shortspokes_r2=0.2
spokes_sep_r2=0.01

##### column threadholdz
th_col=0.004 # how much deviation from mean when identify spokes columns
spokes_sep_c=0.01 # see *spokes_sep_r* for reference
shortspokes_c=0.01 # see *shortspokes_r* for reference


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

# get row numbers with spokes
withspi=getrows(datapoints,th_med,dk_ring_med,thmn,thmx,spokes_sep_r,shortspokes_r)
withspi_1d=[]
withspj_1d=[]

for i in withspi:
	for k in i:
		withspi_1d.append(k)
withspi_1d=np.array(withspi_1d) # put into 1d

# get column numbers with spokes
withspj=getcols(withspi,datapoints,th_col,spokes_sep_c,shortspokes_c)
for i in withspj:
	for k in i:
		for j in k:
			withspj_1d.append(j)
withspj_1d=np.array(withspj_1d) # put into 1d
#print(withspj_1d)


if len(withspi_1d)==0 or len(withspj_1d)==0:
	print("No spokes!")
	exit(1)




################## get image data and save in file  ####################################

filenameimg=filename+'image.txt'
f=open(filenameimg,'w')
f.write('#image size: '+'('+str(m)+','+str(n)+')'+'\n')
f.write('#total number of pixcel:, '+str(totpix)+'\n')
f.write('#minLon maxLon minRad maxRad\n')
f.write(str(a.mnlon)+' '+str(a.mxlon)+' '+str(a.mnrad)+' '+str(a.mxrad)+'\n')
f.write('########################################### below are median intensity for each rad #################################################################\n')
f.write('#Rad Median\n')
	
############################################################################################################



################## plotting original image and smoothed image ####################################
# 2D original plot without spokes 
plt.figure()
plt.title('original image')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')

# smooth background with median 
plt.figure()
plt.title('original image - median of each row')
for i in range(m):
	med=np.median(datapoints[i,:])
	f.write(str(rad_array[i])+' '+str(med)+'\n')
	datapoints[i,:]=datapoints[i,:]-med
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.savefig(filename+'Orig.png')
f.close()
############################################################################################################


# cropped the data so only ones with spokes are left
datacrop=datapoints[np.reshape(withspi_1d,(len(withspi_1d),1)),np.reshape(withspj_1d,(1,len(withspj_1d)))]

# spoke boxes
#print([len(withspj[i]) for i in range(len(withspi))])
spokesnumb=sum([len(withspj[i]) for i in range(len(withspi))]) # spokes number
spkbox_prebpx=[[] for i in range(spokesnumb)] # pixal values
spkbox_ind_lon_prebpx=[[] for i in range(spokesnumb)] # pixal indices in lon
spkbox_ind_rad_prebpx=[[] for i in range(spokesnumb)] # pixal indices in rad
numb=0
for i in range(len(withspi)):
	for j in range(len(withspj[i])):
		#print((float(max(withspi[i])-min(withspi[i])))/float(max(withspj[i][j])-min(withspj[i][j])))
		if (float(max(withspi[i])-min(withspi[i])))/float(max(withspj[i][j])-min(withspj[i][j]))>ctr_r:
			spkbox_prebpx[numb]=datapoints[min(withspi[i]):max(withspi[i]),min(withspj[i][j]):max(withspj[i][j])]
			spkbox_ind_lon_prebpx[numb]=range(min(withspj[i][j]),max(withspj[i][j]))
			spkbox_ind_rad_prebpx[numb]=range(min(withspi[i]),max(withspi[i]))
			numb=numb+1

spkbox_prebpx = [spkbox_prebpx[i] for i in range(len(spkbox_prebpx)) if len(spkbox_prebpx[i])>0]
spkbox_ind_lon_prebpx=[spkbox_ind_lon_prebpx[i] for i in range(len(spkbox_ind_lon_prebpx)) if len(spkbox_ind_lon_prebpx[i])>0]
spkbox_ind_rad_prebpx=[spkbox_ind_rad_prebpx[i] for i in range(len(spkbox_ind_rad_prebpx)) if len(spkbox_ind_rad_prebpx[i])>0]
'''
print(len(spkbox_prebpx[0]))
print(len(spkbox_prebpx[0][0]))
print(len(spkbox_ind_lon_prebpx[0]))
print(len(spkbox_ind_rad_prebpx[0]))
#spokesnumb=len(spkbox_prebpx)
'''
spkbox_prebpx_a=[]
spkbox_ind_lon_prebpx_a=[]
spkbox_ind_rad_prebpx_a=[]
for i in range(len(spkbox_prebpx)):
	#print(spkbox_prebpx[i])
	outi=getrows2(spkbox_prebpx[i],th_med,0,thmn,thmx,spokes_sep_r2,shortspokes_r2)
	#print(outi)
	for j in range(len(outi)):
		#print(min(outi[j]))
		spkbox_prebpx_a.append(spkbox_prebpx[i][min(outi[j]):max(outi[j]),:])
		spkbox_ind_rad_prebpx_a.append(spkbox_ind_rad_prebpx[i][min(outi[j]):max(outi[j])])
		spkbox_ind_lon_prebpx_a.append(spkbox_ind_lon_prebpx[i])

spkbox_prebpx=spkbox_prebpx_a
spkbox_ind_lon_prebpx=spkbox_ind_lon_prebpx_a
spkbox_ind_rad_prebpx=spkbox_ind_rad_prebpx_a
spokesnumb=len(spkbox_prebpx)
'''
print('\n')
print(len(spkbox_prebpx[0]))
print(len(spkbox_prebpx[0][0]))
print(len(spkbox_ind_lon_prebpx[0]))
print(len(spkbox_ind_rad_prebpx[0]))
'''
################## subplot of all the data ################
# plot subplots of all the data
# find min in each row
plt.figure()
plt.title('spoke boxes and spokes before clustering')
spokes_ind_lon_prebpx=[[] for i in range(spokesnumb)] # get spokes center in lon for each spoke
spokes_ind_rad_prebpx=[[] for i in range(spokesnumb)] # get spokes center in rad for each spoke
minidx=[]
subpltn=1 # subplot counts
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, spokesnumb))
for i in range(spokesnumb):
	plt.subplot(1,spokesnumb,subpltn)
	#print(spkbox_prebpx[i])
	plt.imshow(spkbox_prebpx[i], cmap = plt.get_cmap('gray'),origin='upper')
	locallon=[]
	spkbox_prebpx[i]=np.array(spkbox_prebpx[i])

	sort_box=sorted(spkbox_prebpx[i].flatten())
	for k in range(int(len(spkbox_prebpx[i]))):
		normf=len(spkbox_prebpx[i][0])*0.2
		minar=np.where(spkbox_prebpx[i] == (sort_box[int(k*normf)]))
		#print(minar)
		#print(len(spkbox_prebpx[i]))
		#print(len(spkbox_prebpx[i][0]))
		spokes_ind_lon_prebpx[i].append(spkbox_ind_lon_prebpx[i][int(minar[1][0])])
		spokes_ind_rad_prebpx[i].append(spkbox_ind_rad_prebpx[i][int(minar[0][0])])
		plt.plot(minar[1],minar[0],'o',c=colors[i])
			
	
		plt.gca().invert_yaxis()
	subpltn=subpltn+1
	
# plot lines to show which columns and rows are identified as having spokes
plt.figure()
plt.subplot(2,1,1)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
for i in range(len(withspi)):
	for j in range(len(withspj[i])):
		plt.plot([min(withspj[i][j]),max(withspj[i][j]),max(withspj[i][j]),min(withspj[i][j]),min(withspj[i][j])],[min(withspi[i]),min(withspi[i]),max(withspi[i]),max(withspi[i]),min(withspi[i])],'b')

plt.title('before getting rid of boxes')
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()


plt.subplot(2,1,2)
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
for i in range(len(spkbox_ind_rad_prebpx)):
	for j in range(len(spkbox_ind_lon_prebpx)):
		plt.plot([min(spkbox_ind_lon_prebpx[i]),min(spkbox_ind_lon_prebpx[i]),max(spkbox_ind_lon_prebpx[i]),max(spkbox_ind_lon_prebpx[i]),min(spkbox_ind_lon_prebpx[i])],[min(spkbox_ind_rad_prebpx[i]),max(spkbox_ind_rad_prebpx[i]),max(spkbox_ind_rad_prebpx[i]),min(spkbox_ind_rad_prebpx[i]),min(spkbox_ind_rad_prebpx[i])],'b')
plt.title('after getting rid of boxes')
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()

plt.savefig(filename+'lines.png')
# shows plot...
#plt.show()






# check if there are other spokes within one box (clustering)
plt.figure()
plt.title('spokes after clustering')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
distant=[[] for i in range(spokesnumb)]
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

spokes_ind_rad_1d=np.hstack(spokes_ind_rad_prebpx)
spokes_ind_lon_1d=np.hstack(spokes_ind_lon_prebpx)

centers = [spokes_ind_rad_prebpx[0][0], spokes_ind_rad_prebpx[0][0]]
X=np.array(zip(spokes_ind_rad_1d,spokes_ind_lon_1d))
X_bandwitharr=np.array(zip(rad_array,lon_array))
#bandwidth = estimate_bandwidth(X,quantile=qufit)
bandwidth = estimate_bandwidth(X_bandwitharr,quantile=0.01) #estimate bandwidth based on the size of the entire graph
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
	spokes_ind_lon[k]=[int(num) for num in X[my_members, 1]]
	spokes_ind_rad[k]=[int(num) for num in X[my_members, 0]]
	#print(max(spokes_ind_rad[k]))
	spkbox[k]=datapoints[min(spokes_ind_rad[k]):max(spokes_ind_rad[k]),min(spokes_ind_lon[k]):max(spokes_ind_lon[k])]
	
    	cluster_center = cluster_centers[k]
    	plt.plot(X[my_members, 1], X[my_members, 0], 'o')
    	plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()
spokesnumb=n_clusters_
print(str(spokesnumb)+' spokes detected!')
print('length of lon: ',len(spokes_ind_lon[0]))


# get the spokes (check if the surrounding pixals are part of the spokes)
countnew=0

# average increase in pixels
totchange=[totchange_pix+1 for j in range(spokesnumb)]
totchangep=[totchange_pix+1 for j in range(spokesnumb)]

britness=[0 for j in range(spokesnumb)]

spokes_ind_newadd=[[] for i in range(spokesnumb)]
spokes_ind_newadd_p=[[] for i in range(spokesnumb)]
spokes_ind=[[[spokes_ind_lon[i][j],spokes_ind_rad[i][j]] for j in range(len(spokes_ind_rad[i]))] for i in range(spokesnumb)]

	
# 2D original plot with spokes 
plt.figure()
colors = cmap(np.linspace(0, 1, spokesnumb))
plt.title('original image with identified spokes before expanding')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')

for i in range(spokesnumb):
	plt.plot(spokes_ind_lon[i],spokes_ind_rad[i],'o',c=colors[i])
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()

for k in range(spokesnumb):
	spokes_ind_newadd_p[k]=spokes_ind[k]
	britness[k]=np.mean([datapoints[b][a] for a,b in spokes_ind_newadd_p[k]])  # get average brightness of the spoke

britness_m=min(britness)

				

while countnew<iteration and any([totchange[i]>totchange_pix for i in range(spokesnumb)]) and all([totchange[i]<=totpix*totchange_pixmax*4 for i in range(spokesnumb)]): #and any(diff[i]>totchange_pix for i in range(6)):
	totchange=[0 for j in range(spokesnumb)]
	for k in range(spokesnumb):
		if totchangep[k]>totchange_pix and totchangep[k]<totpix*totchange_pixmax:
			if countnew==0:
				spokes_ind_newadd_p[k]=spokes_ind[k]
				#britness[k]=np.mean([datapoints[b][a] for a,b in spokes_ind_newadd_p[k]])  # get average brightness of the spoke
				britness[k]=britness_m
				#print(britness[k])
				#britness[k]=min([datapoints[b][a] for a,b in spokes_ind_newadd_p[k]])  # get min brightness of the spoke
			else:
				spokes_ind_newadd_p[k]=spokes_ind_newadd[k]
				spokes_ind_newadd[k]=[]
			
			orilen=len(spokes_ind[k])
			for i in range(len(spokes_ind_newadd_p[k])):
				x=spokes_ind_newadd_p[k][i][0] # long
				y=spokes_ind_newadd_p[k][i][1] # rad
				
				totchange_med,spokes_ind[k]=updatepix(datapoints,spokes_ind[k],britness[k],pixelcheck,x,y,spoke_pix)	
				
				totchange[k]=totchange[k]+totchange_med		
				spokes_ind_newadd
		spokes_ind_newadd[k]=spokes_ind[k][orilen:len(spokes_ind[k])-1]
			
	print('Total pixal changes per spoke: ',totchange)
	totchangep=totchange
	countnew=countnew+1
	
	
if any([totchange[i]>=totpix*totchange_pixmax*4 for i in range(spokesnumb)]):
	print('Error: Choose a smaller spoke_pix value so result can converage!')
	print('Total change: ',totchange)
	exit(1)		
elif countnew>=iteration:
	print("Warning: Result hasn't converage after "+str(iteration)+" iteration")
else:
	print("Finished identifying spokes")
	
# seperate lon and rad
spokes_ind_lon=[[] for i in range(spokesnumb)]	
spokes_ind_rad=[[] for i in range(spokesnumb)]
	
for i in range(spokesnumb):
	for j in range(len(spokes_ind[i])):
		#print(spokes_ind[i])
		spokes_ind_lon[i].append(spokes_ind[i][j][0])
		spokes_ind_rad[i].append(spokes_ind[i][j][1])

# 2D original plot with spokes 
plt.figure()
colors = cmap(np.linspace(0, 1, spokesnumb))
plt.title('original image with identified spokes -- final')
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
#print('length of lon: ',len(spokes_ind_lon[0]))
for i in range(spokesnumb):
	plt.plot(spokes_ind_lon[i],spokes_ind_rad[i],'o',c=colors[i])
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()
plt.savefig(filename+'.png')

print('writing results to file')
# save into file
filenamesafe=filename+'spokes.txt'
with open(filenamesafe,'w') as f:
	f.write('#rad lon intensity(subtracted median) spokes_number\n')
	for i in range(spokesnumb):
		for j in range(len(spokes_ind_lon[i])):
			f.write(str(rad_array[spokes_ind_rad[i][j]])+' '+str(lon_array[spokes_ind_lon[i][j]])+' '+str(datapoints[spokes_ind_rad[i][j],spokes_ind_lon[i][j]])+' '+str(i+1)+'\n')
	
f.close()
print('finished')
'''
plt.figure()
scx=[]
scy=[]
normlon=[(lon_array[i]-np.median(lon_array))/max(lon_array) for i in range(n)]
normrad=[(rad_array[i]-np.median(rad_array))/max(rad_array) for i in range(m)]
for i in range(n):
	for k in range(m):
		#scx.append(normlon[i]**2+normrad[k]**2)
		scx.append(normlon[i])
		scy.append(datapoints[k,i])

	
plt.scatter(scx,scy,c='b',s=1)

for i in range(spokesnumb):
	for j in range(len(spokes_ind_lon[i])):
		#plt.scatter(normlon[spokes_ind_lon[i][j]]**2+normrad[spokes_ind_rad[i][j]]**2,datapoints[spokes_ind_rad[i][j],spokes_ind_lon[i][j]],c=colors[i],s=1)
		plt.scatter(normlon[spokes_ind_lon[i][j]],datapoints[spokes_ind_rad[i][j],spokes_ind_lon[i][j]],c=colors[i],s=1)
plt.ylabel('pixel brightness')
plt.xlabel('recaled longtitude')
	'''
#plt.show()


