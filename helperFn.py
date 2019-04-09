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

#sys.setrecursionlimit(200000) 

spkcount=1
sharpedge=-2
bound=100
nonspk=-1000
exp_spk=999

# function to crop data and give
def cropdata(datapoints,mxlon,mnlon,mxrad,mnrad):
	m,n=datapoints.shape
	nonzind=np.nonzero(datapoints[0,:])
	lon_array=np.linspace(mnlon,mxlon,n)
	rad_array=np.linspace(mnrad,mxrad,m)
	
	lon_array=lon_array[999:nonzind[0][-1]]
	rad_array=rad_array[200:700]
	return lon_array,rad_array,datapoints[200:700,999:nonzind[0][-1]]

# function to get row numbers with spokes
def getrows(datapoints,th_med,dk_ring_med,thmn,thmx,spokes_sep_r,shortspokes_r):
	withspi=[[] for i in range(100)]
	m,n=datapoints.shape # get the shape of the brightness data
	# if there are more than th amount of pixals darker than 0.95*median then we say spokes are in these rows
	thmni=thmn*m
	thmxi=thmx*m
	count=0
	k=0
	for i in range(m):
		med=np.median(datapoints[i,:])
		#checkmd=min(sum(abs(datapoints[i,:]-th_med*med)>0),sum(datapoints[i,:]<th_med*med)) #might cause problem if more spokes than background...
		checkmd=sum(datapoints[i,:]<th_med*med)
		count=count+1
		#print(med)
		if checkmd>thmni and checkmd<thmxi and med>dk_ring_med:
			if count<m*spokes_sep_r:
				withspi[k].append(i)
			else:
				k=k+1
				withspi[k].append(i)
			count=0
			
	# delete very short spokes (probably not spokes)
	delind=[]
	for i in range(len(withspi)):
		if len(withspi[i])<m*shortspokes_r:
			delind.append(i)
	for i in sorted(delind, reverse=True):
    		del withspi[i]
 
	withspi = np.array([np.array(i) for i in withspi])
	#print(withspi)
	return withspi
	

# function to get column numbers with spokes (only use it after using getrows)
def getcols(withspi,datapoints,th_col,spokes_sep_c,shortspokes_c):

	m,n=datapoints.shape # get the shape of the brightness data
	# compare to the mean for each column to the medien of the columns (might need to check wheather there is spokes in the first column) since the mean should be similar if no spokes
	withspj_all=[]
	
	for e in withspi:
		withspj=[[] for i in range(100)]
		newData=np.array([datapoints[i,:] for i in e])
		medp=np.median([np.mean(newData[:,i]) for i in range(len(newData[0,:]))]) # median of the means for all the columns
		count=0
		k=0
		for j in range(n-1):
			count=count+1
			if (abs(np.mean(newData[:,j+1])-medp)>th_col*medp and np.median(newData[:,j+1])-medp<0): # threadhold for how deviated from mean (only works for dark spokes)
				if count<n*spokes_sep_c:
					withspj[k].append(j+1)
				else:
					k=k+1
					withspj[k].append(j+1)
				count=0
		
		# delete very short spokes (probably not spokes)
		delind=[]		
		for j in range(len(withspj)):
			if len(withspj[j])<n*shortspokes_c:
				delind.append(j)
		for j in sorted(delind, reverse=True):
    			del withspj[j]

		withspj = np.array([np.array(i) for i in withspj])
		withspj_all.append(withspj)
		
	#print(withspj_all)
	return withspj_all

# function that updates pixels
def updatepix(datapoints,spokes_ind,britness,pixelcheck,x,y,spoke_pix):
	totchange=0
	m,n=datapoints.shape
	for np1 in range(pixelcheck):
		if (y+np1+1<m) and ([x,y+np1+1] not in spokes_ind) and (abs(datapoints[y+np1+1][x]-britness)<abs(spoke_pix*britness)):
			spokes_ind.append([x,y+np1+1])
			totchange=totchange+1
		
		if (x+np1+1<n) and ([x+np1+1,y] not in spokes_ind) and (abs(datapoints[y][x+np1+1]-britness)<abs(spoke_pix*britness)):
		 	spokes_ind.append([x+np1+1,y])
		 	totchange=totchange+1
		
		if (x+np1+1<n and y+np1+1<m) and ([x+np1+1,y+np1+1] not in spokes_ind) and (abs(datapoints[y+np1+1][x+np1+1]-britness)<abs(spoke_pix*britness)):
		 	spokes_ind.append([x+np1+1,y+np1+1])
		 	totchange=totchange+1
		
		if (y-np1-1>0) and ([x,y-np1-1] not in spokes_ind) and (abs(datapoints[y-np1-1][x]-britness)<abs(spoke_pix*britness)):
		 	spokes_ind.append([x,y-np1-1])
		 	totchange=totchange+1
		
		if (x-np1-1>0) and ([x-np1-1,y] not in spokes_ind) and (abs(datapoints[y][x-np1-1]-britness)<abs(spoke_pix*britness)):
		 	spokes_ind.append([x-np1-1,y])
		 	totchange=totchange+1
		
		if (x-np1-1>0 and y-np1-1>0) and ([x-np1-1,y-np1-1] not in spokes_ind) and (abs(datapoints[y-np1-1][x-np1-1]-britness)<abs(spoke_pix*britness)):
		 	spokes_ind.append([x-np1-1,y-np1-1])
		 	totchange=totchange+1
	return totchange,spokes_ind


# function to get row numbers with spokes
def getrows2(datapoints,th_med,dk_ring_med,thmn,thmx,spokes_sep_r,shortspokes_r):
	withspi=[[] for i in range(100)]
	m,n=datapoints.shape # get the shape of the brightness data
	# if there are more than th amount of pixals darker than 0.95*median then we say spokes are in these rows
	thmni=thmn*m
	thmxi=thmx*m
	count=0
	k=0
	for i in range(m):
		med=np.median(datapoints[i,:])
		#checkmd=min(sum(abs(datapoints[i,:]-th_med*med)>0),sum(datapoints[i,:]<th_med*med)) #might cause problem if more spokes than background...
		checkmd=sum(datapoints[i,:]<th_med*med)
		count=count+1
		#print(med)
		#if checkmd>thmni and checkmd<thmxi and med>dk_ring_med:
		if checkmd<=thmni or checkmd>=thmxi or med<=dk_ring_med:
			if count<m*spokes_sep_r:
				withspi[k].append(i)
			else:
				k=k+1
				withspi[k].append(i)
			count=0
			
	# delete very short spokes (probably not spokes)
	delind=[]
	for i in range(len(withspi)):
		if len(withspi[i])<m*shortspokes_r:
			delind.append(i)
	for i in sorted(delind, reverse=True):
    		del withspi[i]
 
	withspi = np.array([np.array(i) for i in withspi])
	#print(withspi)
	return withspi

from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
# function to clustering
def cluster_spoke(spokes_ind_rad_1d,spokes_ind_lon_1d,bandwidth):
	# spokes_ind_rad_1d: rad of spokes in 1d array
	# spokes_ind_lon_1d: lon of spokes in 1d array
	# bandwidth: bandwidth for clustering
	centers=[spokes_ind_rad_1d[0],spokes_ind_lon_1d[0]]
	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	X=np.array(zip(spokes_ind_rad_1d,spokes_ind_lon_1d))
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
		cluster_center = cluster_centers[k]
		plt.plot(X[my_members, 1], X[my_members, 0], '.')
		plt.plot(cluster_center[1], cluster_center[0], '.', markerfacecolor=col,markeredgecolor='k', markersize=14)
	plt.gca().invert_yaxis()
	plt.gca().invert_yaxis()
	spokesnumb=n_clusters_
	print(str(spokesnumb)+' spokes detected!')
	print('length of lon: ',len(spokes_ind_lon[0]))
	return spokesnumb, spokes_ind_lon, spokes_ind_rad

# function to smooth data
def smoothdat(datapoints,smooth_pix):
	# datapoints: data
	# smooth_pix: how many pixel to smooth
	m,n=datapoints.shape
	'''
	n_N=int(n/smooth_pix)
	m_N=int(m/smooth_pix)
	datapointsNew=np.zeros([m,n_N])
	spokesdata=np.zeros([m,n])
	for i in range(m):
		med=np.median(datapoints[i,:])
		datapoints[i,:]=[(datapoints[i,j]-med) for j in range(n)]
	minda=(min(datapoints.flatten()))
	for i in range(m):
		datapoints[i,:]=[(datapoints[i,j]+abs(minda)) for j in range(n)]
		datapointsNew[i,:]=np.array([np.mean(datapoints[i,k*smooth_pix:(k+1)*smooth_pix]) for k in range(n_N)])
	for i in range(m):
		med=np.median(datapointsNew[i,:])*thread
		for j in range(len(datapointsNew[i,:])):
			# for boundaries:
			if j+checkNN>len(datapointsNew[i,:]):
				if datapointsNew[i,j]<med and all(datapointsNew[i,j-checkNN:j]<med):
					spokesdata[i,j*smooth_pix:(j+1)*smooth_pix][spokesdata[i,j*smooth_pix:(j+1)*smooth_pix]==0]=spkcount	
			elif j-checkNN<0:
				if datapointsNew[i,j]<med and all(datapointsNew[i,j:j+checkNN]<med):
					spokesdata[i,j*smooth_pix:(j+1)*smooth_pix][spokesdata[i,j*smooth_pix:(j+1)*smooth_pix]==0]=spkcount
			else:		
				if datapointsNew[i,j]<med and (all(datapointsNew[i,j:j+checkNN]<med) or all(datapointsNew[i,j-checkNN:j]<med)):
					spokesdata[i,j*smooth_pix:(j+1)*smooth_pix][spokesdata[i,j*smooth_pix:(j+1)*smooth_pix]==0]=spkcount
				
	'''
	# boxed average
	datapointsNew=np.zeros([m,n])
	spokesdata=np.zeros([m,n])
	minda=(min(datapoints.flatten()))
	for i in range(m):
		datapoints[i,:]=[(datapoints[i,j]+abs(minda)) for j in range(n)]
		datapointsNew[i,smooth_pix:n-smooth_pix-1]=[np.mean(datapoints[i,k-smooth_pix:k+smooth_pix]) for k in range(smooth_pix,n-smooth_pix-1)]
		datapointsNew[i,0:smooth_pix]=[np.mean(datapoints[i,0:k+smooth_pix]) for k in range(0,smooth_pix)]
		datapointsNew[i,n-smooth_pix-1:n-1]=[np.mean(datapoints[i,k-smooth_pix:n-1]) for k in range(n-smooth_pix-1,n-1)]
	
	# smooth in colum direction:
	for i in range(n):
		datapointsNew[smooth_pix:m-smooth_pix-1,i]=[np.mean(datapoints[k-smooth_pix:k+smooth_pix,i]) for k in range(smooth_pix,m-smooth_pix-1)]
		datapointsNew[0:smooth_pix,i]=[np.mean(datapoints[0:k+smooth_pix,i]) for k in range(0,smooth_pix)]
		datapointsNew[m-smooth_pix-1:m-1,i]=[np.mean(datapoints[k-smooth_pix:m-1,i]) for k in range(m-smooth_pix-1,m-1)]
	
	plt.figure()
	plt.subplot(2,1,1)
	plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
	plt.title('original')
	plt.subplot(2,1,2)
	plt.imshow(datapointsNew, cmap = plt.get_cmap('gray'),origin='upper')
	plt.title('smoothed')
	
	return datapointsNew
	

# function to get points with spokes (global row version)
def getspokes_row(datapointsNew,spokesdata,checkNN,thread):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# checkNN: check if it is also dark in this many neighboring pixels
	# thread: only check pixels that have brightness values lower than thread*median 
	m,n=datapointsNew.shape
	for i in range(m):
		med=np.median(datapointsNew[i,:])*thread
		for j in range(len(datapointsNew[i,:])):
			# for boundaries:
			if j+checkNN>len(datapointsNew[i,:]):
				if datapointsNew[i,j]<med and all(datapointsNew[i,j-checkNN:j]<med) and spokesdata[i,j]==0:
					spokesdata[i,j]=spkcount	
			elif j-checkNN<0:
				if datapointsNew[i,j]<med and all(datapointsNew[i,j:j+checkNN]<med) and spokesdata[i,j]==0:
					spokesdata[i,j]=spkcount
			else:		
				if datapointsNew[i,j]<med and (all(datapointsNew[i,j:j+checkNN]<med) or all(datapointsNew[i,j-checkNN:j]<med)) and spokesdata[i,j]==0:
					spokesdata[i,j]=spkcount
				
				
	return spokesdata

# function to get points with spokes (local row version)
from scipy.signal import argrelextrema	
def getspokes_row_l(datapoints,spokesdata,checkNN,thread):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# checkNN: check if it is also dark in this many neighboring pixels
	# thread: only check pixels that have brightness values lower than thread*median 
	m,n=datapoints.shape
	for i in range(m):
		minm=argrelextrema(datapoints[i,:],np.less,order=500)
		if i==300:
			plt.figure()
			plt.plot(datapoints[i,:])
			plt.show()
		spokesdata[i,minm]=100			
	plt.figure()
	plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
	plt.plot(np.where(spokesdata==100)[1],np.where(spokesdata==100)[0], 'ro')
	plt.show()
	return spokesdata

# function to find boundaries
def findbound(spokesdata):
	m,n=spokesdata.shape
	b=0
	for i in range(m):
		# find verticle boundaries
		if i==0 or i==m-1:
			# find horizontal boundaries
			for j in range(n):
				if spokesdata[i,j]==spkcount:
					spokesdata[i,j]=bound
				
		else:
			# find horizontal boundaries
			for j in range(n):
				if j==0 or j==n-1:
					if spokesdata[i,j]==spkcount:
						spokesdata[i,j]=bound
				else:
					if spokesdata[i,j]==spkcount and spokesdata[i+1,j]!=spkcount and spokesdata[i+1,j]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i-1,j]!=spkcount and spokesdata[i-1,j]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i,j-1]!=spkcount and spokesdata[i,j-1]!=bound:
						spokesdata[i,j]=bound
					elif spokesdata[i,j]==spkcount and spokesdata[i,j+1]!=spkcount and spokesdata[i,j+1]!=bound:
						spokesdata[i,j]=bound		
	return spokesdata

	

# function to identify different spoke boundaries after finish identifying spokes boundary:
def findspoke_num(spokesdata,boundsiz,minrowsiz):
	# boundsiz: if the size of the boundary points are less than <boundsiz> then eliminate the spoke
	# minrowsiz: if the row size of the boundary points are less than <minrowsiz> then eliminate the spoke
	spokecount=1
	m,n=spokesdata.shape
	#orginal=len(np.where(spokesdata==bound)[0])
	while bound in spokesdata:
		wherebound=np.where(spokesdata==bound)
		starti=wherebound[0][0]
		startj=wherebound[1][0]

		boundnewzip=[(starti,startj)]
		boundtrack_o=boundnewzip
		
		newb=bound+spokecount # new number for new spoke's boundary
		inc=10
		spokesdata[starti,startj]=newb
		#print(len(wherebound[0]))
		while inc>0:
			inc=0
			boundtrack_n=[]
			for (si,sj) in boundtrack_o:
				if si!=m-1:
					if spokesdata[si+1,sj]==bound:
						spokesdata[si+1,sj]=newb
						boundnewzip.append((si+1,sj))
						boundtrack_n.append((si+1,sj))
						inc=inc+1
				if si!=0:
					if spokesdata[si-1,sj]==bound:
						spokesdata[si-1,sj]=newb
						boundnewzip.append((si-1,sj))
						boundtrack_n.append((si-1,sj))
						inc=inc+1
				if sj!=0:
					if spokesdata[si,sj-1]==bound:
						spokesdata[si,sj-1]=newb
						boundnewzip.append((si,sj-1))
						boundtrack_n.append((si,sj-1))
						inc=inc+1
				if sj!=n-1:
					if spokesdata[si,sj+1]==bound:
						spokesdata[si,sj+1]=newb
						boundnewzip.append((si,sj+1))
						boundtrack_n.append((si,sj+1))
						inc=inc+1
				if si!=m-1 and sj!=n-1:
					if spokesdata[si+1,sj+1]==bound:
						spokesdata[si+1,sj+1]=newb
						boundnewzip.append((si+1,sj+1))
						boundtrack_n.append((si+1,sj+1))
						inc=inc+1
				if si!=0 and sj!=0:
					if spokesdata[si-1,sj-1]==bound:
						spokesdata[si-1,sj-1]=newb
						boundnewzip.append((si-1,sj-1))
						boundtrack_n.append((si-1,sj-1))
						inc=inc+1
				if si!=0 and sj!=n-1:
					if spokesdata[si-1,sj+1]==bound:
						spokesdata[si-1,sj-1]=newb
						boundnewzip.append((si-1,sj+1))
						boundtrack_n.append((si-1,sj+1))
						inc=inc+1
				if si!=m-1 and sj!=0:
					if spokesdata[si+1,sj-1]==bound:
						spokesdata[si+1,sj-1]=newb
						boundnewzip.append((si+1,sj-1))
						boundtrack_n.append((si+1,sj-1))
						inc=inc+1
			boundtrack_o=boundtrack_n

			
		jdel_a=[k[1] for k in boundnewzip]
		idel_a=[k[0] for k in boundnewzip]	
		#plt.figure()
		#plt.subplot(2,1,1)
		#plt.imshow(spokesdata[0:max(idel_a)+10,0:max(jdel_a)+10])
		if (len(boundnewzip)<boundsiz) or ((max(jdel_a)+1-min(jdel_a))/(max(idel_a)+1-min(idel_a)) > minrowsiz):
			spokesdata[idel_a,jdel_a]=0 # need to change to non-spokes... just for visualization (revise)
			for i in range(min(idel_a),max(idel_a)+1):
				ja=[jdel_a[j] for j in range(len(jdel_a)) if (idel_a[j]==i)]
				spokesdata[i,min(ja):max(ja)]=0 # need to change to non-spokes (revise)
				
		else:
			spokecount=spokecount+1
		#plt.subplot(2,1,2)
		#plt.imshow(spokesdata[0:max(idel_a)+10,0:max(jdel_a)+10])
		#plt.show()
	plt.figure()
	plt.imshow(spokesdata)
	#plt.show()
	return spokesdata

'''
# function to get inter from boundaries starting from one point (including boundary conditions...)
def getint_s(spokesdata,checkpoint,bounddata):
	#print(checkpoint)
	m,n=spokesdata.shape
	if checkpoint[0]==0:
		if checkpoint[1]==0:
			print('0,0',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]+1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
				
		elif checkpoint[1]==n-1:
			print('0,n-1',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]+1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
							
		else:
			print('0,any',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata]
			#print(checkpoint)
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						print(checkpoint)
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==3:
							spokesdata[checkpoint[0]+1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==4:
							#print(checkpoint)
							spokesdata[checkpoint[0]+1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
			
			
	elif checkpoint[0]==m-1:
		if checkpoint[1]==0:
			print('m-1,0',checkpoint)
			checklog=[spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]-1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
			
		elif checkpoint[1]==n-1:
			print('m-1,n-1',checkpoint)
			checklog=[spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]-1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
		else:
			print('m-1,any',checkpoint)
			checklog=[spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==3:
							spokesdata[checkpoint[0]-1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==4:
							spokesdata[checkpoint[0]-1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
		
	else:
		#print(m,n)
		#print(checkpoint)
		if checkpoint[1]==0:
			print('any,0',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]+1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==3:
							spokesdata[checkpoint[0]-1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==4:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
				
		elif checkpoint[1]==n-1:
			print('any,n-1',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata]
			#print(checklog)
			if all(checklog):
				return True
			else:
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0]+1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==3:
							spokesdata[checkpoint[0]-1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==4:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
							
		else:
			print('any,any',checkpoint)
			checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata]
			#print(checklog)
			if all(checklog):
				print('pass',checklog)
				return True
			else:
			
				for i in range(len(checklog)):
					if checklog[i]:
						continue
					else:
						#print(i)
						if i==0:
							spokesdata[checkpoint[0]+1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==1:
							spokesdata[checkpoint[0]-1,checkpoint[1]]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==2:
							spokesdata[checkpoint[0],checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==3:
							spokesdata[checkpoint[0],checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0],checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==4:
							spokesdata[checkpoint[0]-1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==5:
							spokesdata[checkpoint[0]-1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]-1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==6:
							spokesdata[checkpoint[0]+1,checkpoint[1]-1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]-1]
							getint_s(spokesdata,checkpoint,bounddata)
						if i==7:
							spokesdata[checkpoint[0]+1,checkpoint[1]+1]=bounddata
							checkpoint=[checkpoint[0]+1,checkpoint[1]+1]
							getint_s(spokesdata,checkpoint,bounddata)
'''
# function to get inter from boundaries starting from one point
def getint_s(spokesdata,checkpoint,bounddata):
	# spokesdata: data that contains spokes/non-spokes information
	# bounddata: the boundary number
	# checkpoint: the first starting point to check
	nwb=bounddata+bound
	m,n=spokesdata.shape
	checklog=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata]
	checklog2=[spokesdata[checkpoint[0]+1,checkpoint[1]]==bounddata or spokesdata[checkpoint[0]+1,checkpoint[1]]==nwb,spokesdata[checkpoint[0]-1,checkpoint[1]]==bounddata or spokesdata[checkpoint[0]-1,checkpoint[1]]==nwb,spokesdata[checkpoint[0],checkpoint[1]+1]==bounddata or spokesdata[checkpoint[0],checkpoint[1]+1]==nwb,spokesdata[checkpoint[0],checkpoint[1]-1]==bounddata or spokesdata[checkpoint[0],checkpoint[1]-1]==nwb,spokesdata[checkpoint[0]-1,checkpoint[1]-1]==bounddata or spokesdata[checkpoint[0]-1,checkpoint[1]-1]==nwb,spokesdata[checkpoint[0]-1,checkpoint[1]+1]==bounddata or spokesdata[checkpoint[0]-1,checkpoint[1]+1]==nwb,spokesdata[checkpoint[0]+1,checkpoint[1]-1]==bounddata or spokesdata[checkpoint[0]+1,checkpoint[1]-1]==nwb,spokesdata[checkpoint[0]+1,checkpoint[1]+1]==bounddata or spokesdata[checkpoint[0]+1,checkpoint[1]+1]==nwb]
	#print('checklog2',checklog2)
	#print('checklog',checklog)
	if any(checklog) or all(checklog2):
		return True
	else:
		if spokesdata[checkpoint[0]+1,checkpoint[1]]!=bounddata:
			spokesdata[checkpoint[0]+1,checkpoint[1]]=nwb
		if spokesdata[checkpoint[0]-1,checkpoint[1]]!=bounddata:
			spokesdata[checkpoint[0]-1,checkpoint[1]]=nwb
		if spokesdata[checkpoint[0],checkpoint[1]+1]!=bounddata:
			spokesdata[checkpoint[0],checkpoint[1]+1]=nwb
		if spokesdata[checkpoint[0],checkpoint[1]-1]!=bounddata:
			spokesdata[checkpoint[0],checkpoint[1]-1]=nwb
		if spokesdata[checkpoint[0]-1,checkpoint[1]-1]!=bounddata:
			spokesdata[checkpoint[0]-1,checkpoint[1]-1]=nwb
		if spokesdata[checkpoint[0]-1,checkpoint[1]+1]!=bounddata:
			spokesdata[checkpoint[0]-1,checkpoint[1]+1]=nwb
		if spokesdata[checkpoint[0]+1,checkpoint[1]-1]!=bounddata:
			spokesdata[checkpoint[0]+1,checkpoint[1]-1]=nwb
		if spokesdata[checkpoint[0]+1,checkpoint[1]+1]!=bounddata:
			spokesdata[checkpoint[0]+1,checkpoint[1]+1]=nwb
		
		if spokesdata[checkpoint[0]+1,checkpoint[1]]!=nwb:
			getint_s(spokesdata,[checkpoint[0]+1,checkpoint[1]],bounddata)
		if spokesdata[checkpoint[0]-1,checkpoint[1]]!=nwb:
			getint_s(spokesdata,[checkpoint[0]-1,checkpoint[1]],bounddata)
		if spokesdata[checkpoint[0],checkpoint[1]+1]!=nwb:
			getint_s(spokesdata,[checkpoint[0],checkpoint[1]+1],bounddata)
		if spokesdata[checkpoint[0],checkpoint[1]-1]!=nwb:
			getint_s(spokesdata,[checkpoint[0],checkpoint[1]-1],bounddata)
		if spokesdata[checkpoint[0]-1,checkpoint[1]-1]!=nwb:
			getint_s(spokesdata,[checkpoint[0]-1,checkpoint[1]-1],bounddata)
		if spokesdata[checkpoint[0]-1,checkpoint[1]+1]!=nwb:
			getint_s(spokesdata,[checkpoint[0]-1,checkpoint[1]+1],bounddata)
		if spokesdata[checkpoint[0]+1,checkpoint[1]-1]!=nwb:
			getint_s(spokesdata,[checkpoint[0]+1,checkpoint[1]-1],bounddata)
		if spokesdata[checkpoint[0]+1,checkpoint[1]+1]!=nwb:
			getint_s(spokesdata,[checkpoint[0]+1,checkpoint[1]+1],bounddata)
					
	
	
# function to fill in boundaries
def getint(spokesdata):
	# spokesdata: data that contains spokes/non-spokes information
	m,n=spokesdata.shape
	boundrange=range(bound+1,int(max(spokesdata.flatten())+1))
	for b in boundrange:
		print(b)
		checkpoints=np.where(spokesdata==b)
		getint_s(spokesdata,[int(np.median(checkpoints[0])),int(np.median(checkpoints[1]))],b)
	plt.figure()
	plt.imshow(spokesdata, cmap = plt.get_cmap('gray'),origin='upper')
	#plt.show()
	

# function to fill in boundaries without recursive function
def getint_nr_s(spokesdata,bounddata):
	# spokesdata: data that contains spokes/non-spokes information
	# bounddata: the boundary number
	m,n=spokesdata.shape
	b=bounddata
	boundrange=range(bound+1,int(max(spokesdata.flatten())+1))
	fb=np.where(spokesdata==b)
	#print(fb)
	fbi=fb[0]
	fbj=fb[1]
	if len(fbi)==0:
		return spokesdata
	coloredfbi=np.array(range(min(fb[0]),max(fb[0])))
	coloredfbj=np.zeros(len(coloredfbi))
	# color the easy ones
	for i in np.unique(fb[0]):
		if i==0 or i==min(fb[0]):
			continue
		else:
			#print(i)
			index_co=np.where(coloredfbi==i)[0]
			ja_pre=[fbj[j] for j in range(len(fbj)) if (fbi[j]==i)]
			if len(ja_pre)==2:
				spokesdata[i,ja_pre[0]+1:ja_pre[1]]=b+bound
				
			else:
				ja_pre=sorted(ja_pre)
				ja=[[ja_pre[j],ja_pre[j+1]] for j in range(len(ja_pre)-1) if ja_pre[j+1]-ja_pre[j]>1] # get gap pairs
				if len(ja)==1:
					spokesdata[i,ja[0][0]+1:ja[0][1]]=b+bound
				else:	
					#print(ja_pre)
					jar=range(min(ja_pre),max(ja_pre)+1)
					finishc=0
					for k in range(len(jar)):
						if spokesdata[i,jar[k]]==spkcount or spokesdata[i,jar[k]]==nonspk or ((spokesdata[i,jar[k]] in boundrange) and spokesdata[i,jar[k]]!=b):
							spokesdata[i,jar[k]] = (b+bound)
					
					
					checkcon=np.zeros(len(jar)) # fill in blanks 
					for k in range(len(jar)):
						if k+1>=len(jar):
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k-1]]==(b+bound):
								checkcon[k]=2
						elif k-1<0:
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k+1]]==(b+bound):
								checkcon[k]=1
						else:
							#print(jar[k])
							#print(jar[k+1])
							if spokesdata[i,jar[k]]==b and spokesdata[i,jar[k+1]]==(b+bound):
								checkcon[k]=1
							elif spokesdata[i,jar[k]]==b and spokesdata[i,jar[k-1]]==(b+bound):
								checkcon[k]=2
					where1=np.where(checkcon==1)
					where2=np.where(checkcon==2)
					
					for k in range(min(len(where1[0]),len(where2[0]))):
						#print(k)
						if len(where1[0])==1:
							spokesdata[i,jar[where1[0][0]]:jar[where2[0][0]]]=b+bound
						else:
							spokesdata[i,jar[where1[0][k]]:jar[where2[0][k]]]=b+bound
							
	return spokesdata
			
# function to fill in boundaires w/o recursion
def getint_nr(spokesdata):
	# spokesdata: data that contains spokes/non-spokes information
	boundrange=range(bound+1,int(max(spokesdata.flatten())+1))
	for b in boundrange:
		getint_nr_s(spokesdata,b)
	return spokesdata


# function to expand small spokes
def expand_spokes(datapoints,spokesdata,spk_num,iteration,brightness,thread,pixelcheck):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# spk_num: the ID number for the spoke
	# iteration: how many iterations to expand
	# brightness: what brightness to compare the pixel value to (normally pix the darkest brightness of all the spokes)
	# thread: if the brighness of the pixals next to that of the darkest spot of the spokes is within *thread* fraction then it is also a spoke pixal
	# pixelcheck: how many neighborning pixels to check
	m,n=datapoints.shape
	totpix=m*n
	where_spk=np.where(spokesdata==spk_num)
	spki=where_spk[0] # i indices
	spkj=where_spk[1] # j indices
	countnew=0
	totchange=20
	changenewi=np.zeros(totpix)-5
	changenewj=np.zeros(totpix)-5
	
	changenewior=np.zeros(totpix)-5
	changenewjor=np.zeros(totpix)-5
	changenewior[0:len(spki)]=spki
	changenewjor[0:len(spkj)]=spkj
	while countnew<iteration and totchange>1: # total change is more than 1 pixel or iteration is smaller than <iteration>
		totchange=0
		changenewi=np.zeros(totpix)-5
		changenewj=np.zeros(totpix)-5
		for i in range(totpix):
			if changenewior[i]==-5:
				break
			else:
				y=int(changenewior[i])
				x=int(changenewjor[i])
				# only checks 4 directions... should be okay
				for np1 in range(pixelcheck):
					if (y+np1+1<m):
						if spokesdata[y+np1+1,x]!=spk_num and spokesdata[y+np1+1,x]!=exp_spk and ((datapoints[y+np1+1][x]-brightness)<(thread*brightness)):
							spokesdata[y+np1+1,x]=exp_spk
							changenewi[totchange]=y+np1+1
							changenewj[totchange]=x
							totchange=totchange+1
					
					if (x+np1+1<n):
						if spokesdata[y,x+np1+1]!=spk_num and spokesdata[y,x+np1+1]!=exp_spk and ((datapoints[y][x+np1+1]-brightness)<(thread*brightness)):
		 					spokesdata[y,x+np1+1]=exp_spk
							changenewi[totchange]=y
							changenewj[totchange]=x+np1+1
		 					totchange=totchange+1
		
					if (y-np1-1>0):
						if spokesdata[y-np1-1,x]!=spk_num and spokesdata[y-np1-1,x]!=exp_spk and ((datapoints[y-np1-1][x]-brightness)<(thread*brightness)):
		 					spokesdata[y-np1-1,x]=exp_spk
							changenewi[totchange]=y-np1-1
							changenewj[totchange]=x
		 					totchange=totchange+1
					
					if (x-np1-1>0):
						if spokesdata[y,x-np1-1]!=spk_num and spokesdata[y,x-np1-1]!=exp_spk and ((datapoints[y][x-np1-1]-brightness)<(thread*brightness)):
		 					spokesdata[y,x-np1-1]=exp_spk
							changenewi[totchange]=y
							changenewj[totchange]=x-np1-1
		 					totchange=totchange+1
		changenewior=changenewi
		changenewjor=changenewj
	return spokesdata
		
		
		
	
	

# function to get rid of sharp edges
def sharpedg(datapoints,spokesdata,thread):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# thread: how much drop is considered a sharp drop
	m,n=datapoints.shape
	for i in range(m):
		if i==0:
			for j in range(n):
				if abs(datapoints[i,j]-datapoints[i+1,j])>thread:
					spokesdata[i,j]=sharpedge
				elif j==0:
					if any([abs(datapoints[i,j]-datapoints[i+1,j+1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread]):
						spokesdata[i,j]=sharpedge
				elif j==n-1:
					if any([abs(datapoints[i,j]-datapoints[i+1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread]):
						spokesdata[i,j]=sharpedge
				else:
					if any([abs(datapoints[i,j]-datapoints[i+1,j+1])>thread,abs(datapoints[i,j]-datapoints[i+1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread]):
						spokesdata[i,j]=sharpedge
		if i==m-1:
			for j in range(n):
				if abs(datapoints[i,j]-datapoints[i-1,j])>thread:
					spokesdata[i,j]=sharpedge
				elif j==0:
					if any([abs(datapoints[i,j]-datapoints[i-1,j+1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread]):
						spokesdata[i,j]=sharpedge
				elif j==n-1:
					if any([abs(datapoints[i,j]-datapoints[i-1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread]):
						spokesdata[i,j]=sharpedge
				else:
					if any([abs(datapoints[i,j]-datapoints[i-1,j+1])>thread,abs(datapoints[i,j]-datapoints[i-1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread]):
						spokesdata[i,j]=sharpedge
		else:
			for j in range(n):
				if any([abs(datapoints[i,j]-datapoints[i-1,j])>thread,abs(datapoints[i,j]-datapoints[i+1,j])>thread]):
					spokesdata[i,j]=sharpedge
				elif j==0:
					if any([abs(datapoints[i,j]-datapoints[i-1,j+1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread,abs(datapoints[i,j]-datapoints[i+1,j+1])>thread]):
						spokesdata[i,j]=sharpedge
				elif j==n-1:
					if any([abs(datapoints[i,j]-datapoints[i-1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread,abs(datapoints[i,j]-datapoints[i+1,j-1])>thread]):
						spokesdata[i,j]=sharpedge
				else:
					if any([abs(datapoints[i,j]-datapoints[i-1,j+1])>thread,abs(datapoints[i,j]-datapoints[i-1,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j-1])>thread,abs(datapoints[i,j]-datapoints[i,j+1])>thread,abs(datapoints[i,j]-datapoints[i+1,j+1])>thread,abs(datapoints[i,j]-datapoints[i+1,j-1])>thread]):
						spokesdata[i,j]=sharpedge
		
			
	return spokesdata
			
			
			
# function to get rid of repeating spokes:
def sortSpk(spokesdata,spkrange):
	# spokesdata: data that contains spokes/non-spokes information
	# spkrange: spoke id numbers to check
	print(spkrange)
	spkboxid=len(spkrange)
	spkbox_i_min=np.zeros(spkboxid)
	spkbox_i_max=np.zeros(spkboxid)
	spkbox_j_min=np.zeros(spkboxid)
	spkbox_j_max=np.zeros(spkboxid)
	spkcount=0
	for spk_id in spkrange:
		wherespk=np.where(spokesdata==spk_id)
		if len(wherespk)==0:
			spkbox_i_min=spkbox_i_min[0:-1]
			spkbox_i_max=spkbox_i_max[0:-1]
			spkbox_j_min=spkbox_j_min[0:-1]
			spkbox_j_max=spkbox_j_max[0:-1]
			spkboxid=spkboxid-1
			spkrange.delete(spk_id)
		else:
			spkbox_i_min[spkcount]=min(wherespk[0])
			spkbox_i_max[spkcount]=max(wherespk[0])
			spkbox_j_min[spkcount]=min(wherespk[1])
			spkbox_j_max[spkcount]=max(wherespk[1])
			spkcount=spkcount+1
	#print('spkbox_i_min',spkbox_i_min)
	#print('spkbox_i_max',spkbox_i_max)
	#print('spkbox_j_min',spkbox_j_min)
	#print('spkbox_j_max',spkbox_j_max)
	
	#print(spkrange)
	for i in range(spkcount):
		#print(i)
		#print([(i!=j and spkbox_i_min[i]>0.8*spkbox_i_min[j] and spkbox_i_max[i]<1.2*spkbox_i_max[j] and spkbox_j_max[i]<1.2*spkbox_j_max[j] and spkbox_j_min[i]>0.8*spkbox_j_min[j]) for j in range(spkcount)])
		if any([(i!=j and spkbox_i_min[i]>0.95*spkbox_i_min[j] and spkbox_i_max[i]<1.05*spkbox_i_max[j] and spkbox_j_max[i]<1.05*spkbox_j_max[j] and spkbox_j_min[i]>0.95*spkbox_j_min[j]) for j in range(spkcount)]):
			#print(spkcount)
			#print(spkrange[i])
			spokesdata[np.where(spokesdata==spkrange[i])]=0
	
	return spokesdata
	
			
			
from scipy import signal
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)			
				
	

