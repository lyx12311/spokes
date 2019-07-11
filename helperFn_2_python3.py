#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Author: Lucy Lu (last update 07/11/2019)
# Contains function libararies for spokes projects. If you have any questions, please reach me via email: lucylulu12311@gmail.com  
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

#sys.setrecursionlimit(200000) 
# these are functions that can be used to find spokes
################################################################################################################################################################
################################################################################################################################################################
#################################################### set ids for different identifications #####################################################################
################################################################################################################################################################
################################################################################################################################################################
# IDs for different identifier in 2d grid
spkcount=1 		# ID for pixels that are identified as spokes but haven't clasify as which spoke it belongs to (ID 2 and up has already been assigned a spoke number)
sharpedge=-2		# Sharp edge pixels when pixel brightness change by too much (not really useful?)
bound=100		# ID for the boundery of spokes
nonspk=-1000		# ID for pixels that are definetly not spokes
exp_spk=999		# ID for pixels that are spokes from expanding but haven't been assigned a spoke number (only use if user expand the spokes)
peaks_ind=500		# ID for pixels that is a peak in that row (only use if user uses gaussian fitting to find spokes)

################################################################################################################################################################
################################################################################################################################################################
#################################################### start of process/smooth image #############################################################################
################################################################################################################################################################
################################################################################################################################################################
# function to crop original data into rectangles, return the new longtitude and radius array as well as the cropped data points (might not need)
def cropdata(datapoints,mxlon,mnlon,mxrad,mnrad):
	# datapoints: original data
	# maxlon: maximum longtitude
	# mnlon: minimum longtitude
	# mxrad: maximum radian
	# mnrad: minimum radian
	m,n=datapoints.shape
	nonzind=np.nonzero(datapoints[0,:])
	lon_array=np.linspace(mnlon,mxlon,n)
	rad_array=np.linspace(mnrad,mxrad,m)
	
	lon_array=lon_array[999:nonzind[0][-1]]
	rad_array=rad_array[200:700]
	return lon_array,rad_array,datapoints[200:700,999:nonzind[0][-1]]

import cv2
# 2d fft to get rid of horizontal noises for each row
def fft2lpf(datapoints,passfiltrow,passfiltcol):
	# passfilt(row/col): how many rows/col to get rid of
	img_float32 = np.float32(datapoints)
	'''
	plt.figure()
	plt.subplot(3,1,1)
	plt.title('original')
	plt.imshow(img_float32, cmap = plt.get_cmap('gray'),origin='upper')
	#plt.show()
	'''
	dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

	rows, cols = datapoints.shape

	# create a mask first, center square is 1, remaining all zeros
	mask = np.zeros((rows, cols, 2), np.uint8)
	mask[0+passfiltrow:rows-passfiltrow, 0+passfiltcol:cols-passfiltcol] = 1
	#print(mask)

	# apply mask and inverse DFT
	fshift = dft*mask
	img_back = cv2.idft(fshift)
	'''
	#plt.figure()
	#print(img_back-img_float32)
	plt.subplot(3,1,2)
	plt.imshow(img_back[:,:,0],cmap = plt.get_cmap('gray'),origin='upper')
	plt.title('after 2d fft')
	#plt.show()
	'''
	return(img_back[:,:,0])

# function to smooth data by mean
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

# function to get rid of bright spots:
#def delbright(datapoints):

# function to smooth data by median (too slow... do not use)
def smoothdat_med(datapoints,smooth_pix):
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
	print('starting row')
	for i in range(m):
		print(i)
		datapoints[i,:]=[(datapoints[i,j]+abs(minda)) for j in range(n)]
		datapointsNew[i,smooth_pix:n-smooth_pix-1]=[np.median(datapoints[i,k-smooth_pix:k+smooth_pix]) for k in range(smooth_pix,n-smooth_pix-1)]
		datapointsNew[i,0:smooth_pix]=[np.median(datapoints[i,0:k+smooth_pix]) for k in range(0,smooth_pix)]
		datapointsNew[i,n-smooth_pix-1:n-1]=[np.median(datapoints[i,k-smooth_pix:n-1]) for k in range(n-smooth_pix-1,n-1)]
	print('finished row')
	# smooth in colum direction:
	print('starting column')
	for i in range(n):
		print(i)
		datapointsNew[smooth_pix:m-smooth_pix-1,i]=[np.median(datapoints[k-smooth_pix:k+smooth_pix,i]) for k in range(smooth_pix,m-smooth_pix-1)]
		datapointsNew[0:smooth_pix,i]=[np.median(datapoints[0:k+smooth_pix,i]) for k in range(0,smooth_pix)]
		datapointsNew[m-smooth_pix-1:m-1,i]=[np.median(datapoints[k-smooth_pix:m-1,i]) for k in range(m-smooth_pix-1,m-1)]
	
	plt.figure()
	plt.subplot(2,1,1)
	plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
	plt.title('original')
	plt.subplot(2,1,2)
	plt.imshow(datapointsNew, cmap = plt.get_cmap('gray'),origin='upper')
	plt.title('smoothed')
	
	return datapointsNew

from scipy import signal
# used in blur_image(im, n, ny=None)
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

# gaussian kernel smoothing image
def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)			
				
	
from scipy.signal import butter, lfilter, freqz
# used in reduce_highFn(datapoints,cutoff=0.05)
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# used in reduce_highFn(datapoints,cutoff=0.05)
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
   
# low pass filter for image
def reduce_highFn(datapoints,cutoff=0.05):
	order = 1
	fs = 1.0       # sample rate, Hz
	#cutoff=0.005
	#cutoff=0.012
	m,n=datapoints.shape
	artdat=90
	for i in range(m):
		findat=butter_lowpass_filter(np.append(np.ones(artdat)*datapoints[i,0],datapoints[i,:]), cutoff, fs, order)
		datapoints[i,:] = findat[artdat:len(findat)]
	for j in range(n):
		findat=butter_lowpass_filter(np.append(np.ones(artdat)*datapoints[0,j],datapoints[:,j]), cutoff, fs, order)
		datapoints[:,j] = findat[artdat:len(findat)]
	return datapoints

# function to get rid of sharp edges (doesn't really work...)
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
################################################################################################################################################################
################################################################################################################################################################		
############################################# end of process/smooth image ######################################################################################
################################################################################################################################################################
################################################################################################################################################################

################################################################################################################################################################
################################################################################################################################################################
########################################### start of getting spokes pixels and set them to <spkcount> ##########################################################
################################################################################################################################################################
################################################################################################################################################################
# function to get spoke pixels by comparing the median for each row
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

# get darkest pixels
def getdark(datapoints,spokesdata,darkpix):
	#darkpix: how many dark pixels to get to count as spokes
	m,n=datapoints.shape
	dataflat=datapoints.flatten()
	data,index=list(zip(*sorted(zip(dataflat,list(range(len(dataflat)))))))
	#print([index[0:darkpix]])
	for i in range(len(data)):
		#print(data[i])
		if data[i]>darkpix:
			break
			
	#print(i)
	back2d=np.unravel_index([index[0:i]],(m,n)) # getting the index back to 2d
	spokesdata[back2d]=spkcount
	#print(back2d)
	#plt.plot(back2d[1],back2d[0],'r.')
	return(spokesdata)
	

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# function to get spoke pixels by peak finding (gaussian method)
def getspokes_row2(datapoints,spokesdata,prom=0.05,width_s=[1,1000],pw=0):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# prom: Required prominence of peaks, only take <prom> fraction of the data.
	# width_s: Required width range of peaks in samples. 
	# rel_height_s: Chooses the relative height at which the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at least 0. See notes for further explanation.
	# pw: mark the whole peak as spokes based on the width from find peak, 0 means no and 1 means yes
	prex=np.zeros(200)
	aftx=np.zeros(200)
	m,n=datapoints.shape
	'''
	plt.figure()
	plt.subplot(2,1,1)
	plt.imshow(datapoints,cmap = plt.get_cmap('gray'),origin='upper')
	plt.subplot(2,1,2)
	plt.imshow(datapoints,cmap = plt.get_cmap('gray'),origin='upper')
	'''
	print('getting data')
	widths=[[] for i in range(m)]
	width_heights=[[] for i in range(m)]
	prominences=[[] for i in range(m)]
	peaks_ar=[[] for i in range(m)]
	for i in range(m):
		prex[-1]=-datapoints[i,0]-(-datapoints[i,0]-min(-datapoints[i,:]))/len(prex)
		aftx[0]=-datapoints[i,-1]-(-datapoints[i,-1]-min(-datapoints[i,:]))/len(prex)
		for j in range(1,len(prex)):
			prex[len(prex)-j-1]=prex[-1]-(prex[-1]-min(-datapoints[i,:]))/len(prex)*j
			aftx[j]=aftx[0]-(aftx[0]-min(-datapoints[i,:]))/len(prex)*j
		'''
		plt.figure()
		plt.plot(-datapoints[i,:])
		plt.show()
		'''
		x=np.append(np.append(prex,-datapoints[i,:]),aftx)
		peaks, dicts = find_peaks(x,prominence=0.0,width=1)
		#print(dicts)
		widths[i]=dicts['widths']
		width_heights[i]=dicts['width_heights']
		prominences[i]=dicts['prominences']
		peaks_ar[i]=peaks-len(prex)
		
	
		#plt.show()
		#print(max(prominences))
		#print(np.median(prominences))
	print('finished')
	
	prominencesf= []
	width_heightsf=[]
	for i in prominences:
		for j in i:
			prominencesf.append(j)
		
	for i in width_heights:
		for j in i:
			width_heightsf.append(j)
	
	#plt.figure()
	counts, bins =np.histogram(prominencesf,100,normed=False)
	#plt.figure()
	#plt.plot(counts)
	for i in range(len(counts)):
		if counts[i]<prom*counts[0]:
			break
	#plt.title('prominences')
	#plt.yscale('log')
	'''	
	plt.figure()
	plt.semilogx(prominencesf,width_heightsf,'.')
	plt.plot(np.median(prominencesf),np.median(width_heightsf),'o')
	#print((np.median(prominences),np.median(width_heights)))
	plt.xlabel('prominences')
	plt.ylabel('heights')
	plt.plot([bins[i],bins[i]],[min(width_heightsf),max(width_heightsf)],'r-')
	#print(bins[i])
	plt.show()
	'''
	checkp=bins[i]
	print('sorting data')
	for i in range(m):
		for ind in reversed(list(range(len(widths[i])))):
			if width_heights[i][ind]<-0.06 or prominences[i][ind]<checkp or widths[i][ind]>width_s[1] or widths[i][ind]<width_s[0]:
				np.delete(peaks_ar[i],ind)
			else:
				spokesdata[i,peaks_ar[i][ind]]=spkcount
				
				if pw==1:
					leftid=peaks_ar[i][ind]-widths[i][ind]/5.
					rightid=peaks_ar[i][ind]+widths[i][ind]/5.
					if leftid<0:
						spokesdata[i,0:rightid]=spkcount
					elif rightid>n-1:
						spokesdata[i,leftid:n-1]=spkcount
					else:
						spokesdata[i,leftid:rightid]=spkcount
	
		
		'''
		if i==200:
			plt.plot([(k-len(prex)) for k in peaks],i*np.ones(len(peaks)),'r.')
			plt.figure()
			plt.plot(x)
			plt.plot(peaks, x[peaks], "x")
			print(dicts)
			#print(dicts['widths'])
			#print(dicts['width_heights'])
			#plt.show()
		
		'''
	print('finished')
	whereii,wherejj=np.where(spokesdata==spkcount)				
	plt.plot(wherejj,whereii,'r.')
	#print('whereii',whereii)
	#print('wherejj',wherejj)
	
		
	'''
	plt.title('peaks')
	plt.figure()
	plt.hist(widths,bins=np.arange(200))
	plt.title('width')
	plt.figure()
	plt.hist(width_heights,bins=np.linspace(-0.022,-0.01,100))
	plt.title('width_heights')
	plt.figure()
	plt.hist(left_ips,bins=np.linspace(0,2100,100))
	plt.title('left_ips')
	plt.figure()
	plt.hist(right_ips,bins=np.linspace(0,2250,100))
	plt.title('right_ips')
	plt.figure()
	plt.hist(prominences,bins=np.linspace(0,0.008,500))
	plt.title('prominences')
	plt.yscale('log')
	plt.figure()
	plt.hist([abs(left_bases[i]-right_bases[i]) for i in range(len(left_bases))],bins=np.linspace(0,2000,100))
	plt.title('left_bases-right')
	'''
	
	
	
	'''
	plt.figure()
	plt.loglog(prominences,widths,'.')
	plt.xlabel('prominences')
	plt.ylabel('width')
	'''
	'''
	plt.figure()
	plt.imshow(spokesdata)
	plt.show()
	'''
	return spokesdata

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
# gaussian function
def gaus(x,a,x0,sigma): # define gaussian function
    return abs(a)*exp(-(x-x0)**2./(2.*sigma**2.))
    
# get the width of peaks by fitting gaussian (the fitting end points are points when the data stops decreasing from the peak, might have a better way...)
def peakwidth(datapoints,spokesdata,peaknumb):
	m,n=datapoints.shape
	peaki,peakj=np.where(spokesdata==peaknumb)
	#print(peaki)
	peakbright_ar=datapoints[peaki,peakj]
	checki=list(range(min(peaki),max(peaki)+1)) # how many rows it covered
	for i in range(len(checki)):
		#print(checki[i])
		ind_tot=np.where(peaki==checki[i])
		indi=ind_tot[0]
		#print(indi)
		# get peak index
		if len(indi)==0:
			x=list(range(indj-leftlen,indj+rightlen))
			indj=np.where(datapoints[checki[i],x]==min(datapoints[checki[i],x]))[0]
			if len(indj)!=1:
				indj=int(indj[0])
			else:
				indj=int(indj)
		else:
			indj_pre=peakj[np.where(peaki==checki[i])[0]]
			if len(indj_pre)==1:
				indj=int(indj_pre)
			else:
				datapeak=datapoints[np.ones(len(indj_pre))*indi,indj_pre] # find out the mins and which one is the min of the mins
				ind,_=np.where(datapeak==min(datapeak))
				indj=int(indj_pre[ind])
				
			
			# get the width to fit gaussian by checking when its increasing
			leftlen=0
			rightlen=0
			#print(len(datapoints[checki[i],0:indj]))
			inc=0
			for j in range(1,len(datapoints[checki[i],0:indj])): # left side
				if -datapoints[checki[i],indj-j]<-datapoints[checki[i],indj-j+1]:
					inc=0
					leftind=indj-j
					leftlen=leftlen+1
					
					#if (leftlen/100.)==int(leftlen/100.):
					#	plt.figure()
					#	plt.plot(-datapoints[checki[i],leftind:indj+1])
					#	plt.show()
					
				else:
					break
			inc=0
			for j in range(indj,n-1): # right side
				#print(j)
				#print(checki[i])
				if -datapoints[checki[i],j+1]<-datapoints[checki[i],j]:
					inc=0
					rightind=j+1
					rightlen=rightlen+1
				else:
					break
					
			x=list(range(indj-leftlen,indj+rightlen))
			'''
			sidelen=max(rightlen,leftlen)
			
			if sidelen<20 and indj+20<n-1 and indj-20>0:
				x=range(indj-20,indj+20)
			elif indj+sidelen>n-1:
				x=range(indj-sidelen,n-1)
			elif indj-sidelen<0:
				x=range(0,indj+sidelen)
			else:
				x=range(indj-sidelen,indj+sidelen)
			'''
			#x=range(indj-20,indj+20)
			
		
		# fit gaussian for each peak
		p0=[float(abs(-datapoints[checki[i],indj]-min(-datapoints[checki[i],x]))),float(indj),float(0.3*(max(x)-min(x)))]
		#print(p0)
		'''
		#if checki[i]==125 or checki[i]==160:
		plt.figure()
		plt.plot(x,-datapoints[checki[i],x]-min(-datapoints[checki[i],x]),'b+:',label='data')
		plt.plot(indj,-datapoints[checki[i],indj]-min(-datapoints[checki[i],x]),'o')
		plt.plot(x,gaus(np.array(x),*p0),'b--',label='initial')
		popt,pcov = curve_fit(gaus,x,-datapoints[checki[i],x]-min(-datapoints[checki[i],x]),p0, maxfev=10000)
		print(popt)
		plt.plot(x,gaus(x,*popt),'r-',label='fit')
		plt.title(str(checki[i]))
		plt.legend()
		plt.show()
		'''
		'''
		plt.figure()
		plt.plot(x,-datapoints[checki[i],x],'b+:',label='data')
		plt.plot(indj,-datapoints[checki[i],indj],'o')
		plt.plot(x,gaus(np.array(x),*p0)+min(-datapoints[checki[i],x]),'b--',label='initial')
		#plt.show()
		popt,pcov = curve_fit(gaus,x,-datapoints[checki[i],x]-min(-datapoints[checki[i],x]),p0, maxfev=10000)
		#print(pcov)
		plt.plot(x,gaus(x,*popt)+min(-datapoints[checki[i],x]),'r-',label='fit')
		plt.legend()
		plt.title(str(checki[i]))
		plt.show()
		'''
		
		
			
		try:
			popt,pcov = curve_fit(gaus,x,-datapoints[checki[i],x]-min(-datapoints[checki[i],x]),p0,maxfev=10000)
		except:
			pass
		
		'''
		if abs(p0[1]-popt[1])<100:
			width=popt[-1]
			spokesdata[checki[i],int(popt[1])-int(width):int(popt[1])+int(width)]=peaknumb
			#plt.plot([int(p0[1])-int(width),int(p0[1])+int(width)],[checki[i],checki[i]],'b.')
			plt.plot([int(popt[1])-int(width),int(popt[1])+int(width)],[checki[i],checki[i]],'b.',markersize=2)
			plt.plot(popt[1],checki[i],'r.',markersize=2
		'''
		width=popt[-1]
		spokesdata[checki[i],int(popt[1])-int(width):int(popt[1])+int(width)]=peaknumb
		#plt.plot([int(p0[1])-int(width),int(p0[1])+int(width)],[checki[i],checki[i]],'b.')
		plt.plot([int(popt[1])-int(width),int(popt[1])+int(width)],[checki[i],checki[i]],'b.',markersize=2)
		plt.plot(popt[1],checki[i],'r.',markersize=2)
		
	return spokesdata
		
		
	#plt.show()
################################################################################################################################################################
################################################################################################################################################################	
########################################### end of getting spokes pixels and set them to <spkcount> ############################################################
################################################################################################################################################################
################################################################################################################################################################

################################################################################################################################################################
################################################################################################################################################################	
########################################### start of getting rid of short spokes/ identify spoke numbers #######################################################
################################################################################################################################################################
################################################################################################################################################################
# function to get rid of single peaks (if no pixel in the neighboring rows that are <extend> away)
def clean_single(spokesdata,extend=5,checkvalue=spkcount):
	# spokesdata: data that contains spokes/non-spokes information
	# extend: how many pixels to check connection in vertical
	# checkvalue: values to check
	m,n=spokesdata.shape
	indi,indj=np.where(spokesdata==checkvalue)
	for i in range(len(indi)):	
		if indi[i]==m-1:
			findi=np.append(np.where(indi==indi[i]-1), np.where(indi==indi[i]))
		elif indi[i]==0:
			findi=np.append(np.where(indi==indi[i]+1), np.where(indi==indi[i]))
		else:
			findi=np.append((np.append(np.where(indi==indi[i]+1), np.where(indi==indi[i]-1))),np.where(indi==indi[i]))
			
		#print(findi)
		if indj[i]==n-extend:
			checkj=[(indj[k]<indj[i] and indj[k]>indj[i]-extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				spokesdata[indi[i],indj[i]]=0
		elif indj[i]==0:
			checkj=[(indj[k]>indj[i] and indj[k]<indj[i]+extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				spokesdata[indi[i],indj[i]]=0
		else:
			checkj=[(indj[k]<indj[i]+extend and indj[k]>indj[i]-extend) for k in findi]
			if any(checkj):
				if sum(checkj)==1:
					spokesdata[indi[i],indj[i]]=0
				else:
					continue
			else:
				#print(spokesdata[indi[i],indj[i]])
				spokesdata[indi[i],indj[i]]=0
	
	return spokesdata
		 
# function to get rid of short peaks and identify different spokes (similar to clean_single but clean anything thats less than <ss> rows and identify different peak IDs)
def clean_short(spokesdata,ss,extend=5):
	# extend: see clean_single() function		
	# ss: how many pixels to consider as short spokes
	m,n=spokesdata.shape
	indi,indj=np.where(spokesdata==spkcount) 
	skc=2
	while len(indi)>0:
		spokesdata[indi[0],indj[0]]=skc
		spki=[indi[0]]
		spkj=[indj[0]]
		spkinc_i=[indi[0]]
		spkinc_j=[indj[0]]
		spkinc_i_old=[]
		spkinc_j_old=[]
		inc=20
		while inc>0:
			inc=0
			for i in range(len(spkinc_i)):
				# get i's
				if spkinc_i[i]==m-1:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]-1) for k in indi]
				elif spkinc_i[i]==0:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]+1) for k in indi]
				else:
					findi=[(k==spkinc_i[i] or k==spkinc_i[i]+1 or k==spkinc_i[i]-1) for k in indi]
				checki=indi[findi]
				checkj=indj[findi]
				
				# check j's
				for j in range(len(checkj)):
					if (spokesdata[checki[j],checkj[j]]==spkcount) and checkj[j]<spkinc_j[i]+extend and checkj[j]>spkinc_j[i]-extend:
						spokesdata[checki[j],checkj[j]]=skc
						spki.append(checki[j])
						spkj.append(checkj[j])
						spkinc_i_old.append(checki[j])
						spkinc_j_old.append(checkj[j])
						inc=inc+1
			spkinc_i=spkinc_i_old
			spkinc_j=spkinc_j_old
			spkinc_i_old=[]	
			spkinc_j_old=[]
		#print(spkj,spki)
		if len(spki)<ss:
			spokesdata[spki,spkj]=0
		else:
			plt.plot(spkj,spki,'.')
		skc=skc+1
		indi,indj=np.where(spokesdata==spkcount)
	return spokesdata	

# function to connect peaks if they are close to each other and get rid of the short peaks (only works if you have the peaks identified using clean_short [using gaussian method])
def connectline(spokesdata,ss,ss_h,spokesarr,extend=10,extend_h=50):	
	# spokesarr: spokes id to check
	# ss: how many pixels to consider as short spokes in rows
	# ss_h: how many pixels to consider as short spokes in columns
	# extend: how many rows to connect	
	# extend_h: how many columns to connect
	where_spk_i=[]
	where_spk_j=[]
	spk_ind=[]
	for i in spokesarr:
		where_spk_i_s,where_spk_j_s=np.where(spokesdata==i)
		if len(where_spk_i_s)==0:
			continue
			
		where_spk_i_s,where_spk_j_s=list(zip(*sorted(zip(where_spk_i_s,where_spk_j_s))))
		where_spk_i.append(where_spk_i_s)
		where_spk_j.append(where_spk_j_s)
		spk_ind.append(i)
	spc=2
	for i in range(len(where_spk_i)):
		#plt.plot(where_spk_j[i],where_spk_i[i],'.')
		for j in range(len(where_spk_i)):
			overlapar=len(set(where_spk_i[i]) & set(where_spk_i[j]))
			#print(overlapar)
			if j==i or overlapar>2:
				continue
			minii=min(where_spk_i[i])
			maxij=max(where_spk_i[j])
			
			minij=min(where_spk_i[j])
			maxii=max(where_spk_i[i])
			
			indminii=np.where(where_spk_i[i]==min(where_spk_i[i]))[0]
			indmaxii=np.where(where_spk_i[i]==max(where_spk_i[i]))[0]
			indminij=np.where(where_spk_i[j]==min(where_spk_i[j]))[0]
			indmaxij=np.where(where_spk_i[j]==max(where_spk_i[j]))[0]
			
			if len(indminii)==1:
				minji=where_spk_j[i][int(indminii)]
			else:
				minji=where_spk_j[i][int(indminii[0])]
			
			if len(indmaxii)==1:
				maxji=where_spk_j[i][int(indmaxii)]
			else:
				maxji=where_spk_j[i][int(indmaxii[0])]
			
			if len(indminij)==1:
				minjj=where_spk_j[j][int(indminij)]
			else:
				minjj=where_spk_j[j][int(indminij[0])]
				
			if len(indmaxij)==1:
				maxjj=where_spk_j[j][int(indmaxij)]
			else:
				maxjj=where_spk_j[j][int(indmaxij[0])]
			
			if minii>maxij and minii-maxij<extend:
				#print(minii,maxij)
				if abs(minji-maxjj)<extend_h:
					indn=min(spk_ind[i],spk_ind[j])
					spokesdata[where_spk_i[i],where_spk_j[i]]=indn
					spokesdata[where_spk_i[j],where_spk_j[j]]=indn
					spk_ind[i]=indn
					spk_ind[j]=indn
					'''
					incr_j=float(minji-maxjj)/float(minii-maxij)
					for k in range(maxij+1,minii):
						spokesdata[k,minij+int(incr_j)*(k-maxij)]=indn
					'''
					whereind=np.where(spokesdata==indn)
					#plt.plot(whereind[1],whereind[0],'.')
			
			elif minij>maxii and minij-maxii<extend:
				if abs(maxji-minjj)<extend_h:
					indn=min(spk_ind[i],spk_ind[j])
					spokesdata[where_spk_i[i],where_spk_j[i]]=indn
					spokesdata[where_spk_i[j],where_spk_j[j]]=indn
					spk_ind[i]=indn
					spk_ind[j]=indn
					'''
					incr_j=float(maxji-minjj)/float(minij-maxii)
					for k in range(maxii+1,minij):
						spokesdata[k,minii+int(incr_j)*(k-maxii)]=indn
					'''
					whereind=np.where(spokesdata==indn)
					#plt.plot(whereind[1],whereind[0],'.')
				
	indnew=np.unique(spk_ind)
	#print(indnew)
	nspk=0
	i=0
	for i in range(len(indnew)):
		#print(i)
		spi,pij=np.where(spokesdata==indnew[i])
		if abs(max(spi)-min(spi))<ss or abs(max(pij)-min(pij))<ss_h:
			nspk=nspk+1
			spokesdata[spi,pij]=0
		else:
			#print('nspk',nspk)
			#print('i',i)
			#print('2+i-nspk',2+i-nspk)
			spokesdata[spi,pij]=2+i-nspk
			plt.plot(pij,spi,'.')
	if i==nspk:
		print(('spokes No.',0))		
	else:
		print(('spokes No.',2+i-nspk-1))
	return spokesdata							
							

# function to clean up very long rows based on the median [truncate very long rows]
def clean_rows(spokesdata,idel_a,jdel_a,colum_length,spkc):
	#print('cleaning')
	# idel_a: i indeices for the spokes boundary (sorted)
	# jdel_a: j indeices for the spokes boundary (sorted)
	# colum_length: how many columns there are for each row (same order as idel_a)
	# spkc: spokes id number
	m,n=spokesdata.shape
	thread=0.2 # how much longer to take action
	rows_s=list(Counter(idel_a).keys())
	colum_length_med=np.median(colum_length)
	for i in range(len(rows_s)):
		rownumbers=[int(ri) for ri in (np.where(idel_a==rows_s[i])[0])]
		wherearerow=jdel_a[min(rownumbers):max(rownumbers)+1]
		if abs(colum_length[i]-colum_length_med)>thread*abs(colum_length_med):
			if i==0 or i==len(rows_s)-1:	
				spokesdata[rows_s[i],min(wherearerow):max(wherearerow)+1]=0
				if int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med)<0:
					spokesdata[rows_s[i],0:int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)]=spkc
				elif int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)>n-1:
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med):n]=spkc
				else:
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med):int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)]=spkc
				
			else:
				spokesdata[rows_s[i],min(wherearerow):max(wherearerow)+1]=0	
				if int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med)<0:
					spokesdata[rows_s[i],0]=spkc
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)]=spkc
				elif int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)>n-1:
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med)]=spkc
					spokesdata[rows_s[i],n-1]=spkc
				else:
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)-int(0.5*colum_length_med)]=spkc
					spokesdata[rows_s[i],int((min(wherearerow)+max(wherearerow))/2.)+int(0.5*colum_length_med)]=spkc
	return spokesdata
################################################################################################################################################################
################################################################################################################################################################
########################################### end of getting rid of short spokes #################################################################################
################################################################################################################################################################
################################################################################################################################################################



################################################################################################################################################################
################################################################################################################################################################
########################################### start of getting spokes boundaries and identify spokes number ######################################################
################################################################################################################################################################
################################################################################################################################################################
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
	
from collections import Counter
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

		#indices for i and j for this spoke	
		jdel_a=[k[1] for k in boundnewzip]
		idel_a=[k[0] for k in boundnewzip]
		# find min and max for each row
		idel_a,jdel_a=list(zip(*sorted(zip(idel_a,jdel_a))))
		
		
		rows_s=list(Counter(idel_a).keys())
		colums_s=list(Counter(idel_a).values())
		# get how many columns are for each row
		for rowi in range(len(rows_s)):
			rownumbers=[int(ri) for ri in (np.where(idel_a==rows_s[rowi])[0])]
			#print(idel_a[min(rownumbers):max(rownumbers)+1])
			wherearerow=jdel_a[min(rownumbers):max(rownumbers)+1]
			colums_s[rowi]=(max(wherearerow)-min(wherearerow))+1
		if (len(boundnewzip)<boundsiz) or ((np.median(colums_s))/(len(rows_s)) > minrowsiz) or len(rows_s)<2:
			spokesdata[idel_a,jdel_a]=0 # need to change to non-spokes... just for visualization (revise)
			for i in range(min(idel_a),max(idel_a)+1):
				ja=[jdel_a[j] for j in range(len(jdel_a)) if (idel_a[j]==i)]
				spokesdata[i,min(ja):max(ja)]=0 # need to change to non-spokes (revise)
				
		else:
			#spokesdata=clean_rows(spokesdata,idel_a,jdel_a,colums_s,newb)
			spokecount=spokecount+1
			
	#plt.figure()
	#plt.imshow(spokesdata)
	#plt.title('after identifying spokes')
	return spokesdata	

################################################################################################################################################################
################################################################################################################################################################
########################################### end of getting spokes boundaries and identify spokes number ########################################################
################################################################################################################################################################
################################################################################################################################################################

################################################################################################################################################################
################################################################################################################################################################
########################################### start of getting filling in spokes based on identified boundaries ##################################################
################################################################################################################################################################
################################################################################################################################################################
# function to get the inside of a spokes from boundaries starting from one point [recursive... fails for big spokes]
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
		return spokesdata
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
					
# function to fill in boundaries (used in getint_s(spokesdata,checkpoint,bounddata)) [recursive]
def getint(spokesdata):
	# spokesdata: data that contains spokes/non-spokes information
	m,n=spokesdata.shape
	boundrange=list(range(bound+1,int(max(spokesdata.flatten())+1)))
	for b in boundrange:
		#print(b)
		checkpoints=np.where(spokesdata==b)
		getint_s(spokesdata,[int(np.median(checkpoints[0])),int(np.median(checkpoints[1]))],b)
	plt.figure()
	plt.imshow(spokesdata, cmap = plt.get_cmap('gray'),origin='upper')
	#plt.show()
	

# function to fill in boundaries without recursive function (recursive function doesn't work well in python...)
def getint_nr_s(spokesdata,bounddata):
	# spokesdata: data that contains spokes/non-spokes information
	# bounddata: the boundary number
	m,n=spokesdata.shape
	b=bounddata
	boundrange=list(range(bound+1,int(max(spokesdata.flatten())+1)))
	fb=np.where(spokesdata==b)
	#print(fb)
	fbi=fb[0]
	fbj=fb[1]
	if len(fbi)==0:
		return spokesdata
	coloredfbi=np.array(list(range(min(fb[0]),max(fb[0]))))
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
				#print(ja)
				if len(ja)==1:
					spokesdata[i,ja[0][0]+1:ja[0][1]]=b+bound
				else:	
					#print(ja_pre)
					jar=list(range(min(ja_pre),max(ja_pre)+1))
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
							for l in range(jar[where1[0][k]],jar[where2[0][k]]):
								if spokesdata[i,l]==spkcount:
									spokesdata[i,l]=b+bound
							
	return spokesdata
			
# function to fill in boundaires w/o recursion
def getint_nr(spokesdata):
	# spokesdata: data that contains spokes/non-spokes information
	boundrange=list(range(bound+1,int(max(spokesdata.flatten())+1)))
	for b in boundrange:
		getint_nr_s(spokesdata,b)
	return spokesdata
################################################################################################################################################################
################################################################################################################################################################
########################################### end of getting filling in spokes based on identified boundaries ####################################################
################################################################################################################################################################
################################################################################################################################################################

################################################################################################################################################################
################################################################################################################################################################
########################################### start of expanding spokes based on origional spokes ################################################################
################################################################################################################################################################
################################################################################################################################################################
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
		countnew=countnew+1
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
		
		
# function to expand small spokes in rows
def expand_spokes_row(datapoints,spokesdata,spk_num,iteration,thread,pixelcheck,brightref):
	# datapoints: data
	# spokesdata: data that contains spokes/non-spokes information
	# spk_num: the ID number for the spoke
	# iteration: how many iterations to expand
	# thread: if the brighness of the pixals next to that of the darkest spot of the spokes is within *thread* fraction then it is also a spoke pixal
	# pixelcheck: how many neighborning pixels to check
	# brightref: if lower than this brightness then not spokes 
	m,n=datapoints.shape
	totpix=m*n
	where_spk=np.where(spokesdata==spk_num)
	brightness_ar=datapoints[where_spk] #brightness of the known pixels
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
	
	Totchange_sum=0
	while countnew<iteration and totchange>1 and Totchange_sum<0.02*totpix: # total change is more than 1 pixel or iteration is smaller than <iteration>
		#print(Totchange_sum)
		countnew=countnew+1
		totchange=0
		changenewi=np.zeros(totpix)-5
		changenewj=np.zeros(totpix)-5
		for i in range(totpix):
			if changenewior[i]==-5:
				break
			else:
				# which point to check
				y=int(changenewior[i])
				x=int(changenewjor[i])
				# check which brighness point is this pixel closes to (row wise)
				diffi=[abs(indb-y) for indb in spki]
				wheremin=np.where(diffi==min(diffi))[0] # where is the pixel that is closest to
				if len(wheremin)==1:
					brightness=brightness_ar[np.where(diffi==min(diffi))[0]]
				else:
					brightness=brightness_ar[np.where(diffi==min(diffi))[0][0]]
				# only checks 4 directions... should be okay
				for np1 in range(pixelcheck):
					if (y+np1+1<m) and min([abs(y+np1+1-z) for z in spki])<0.5*m:
						if spokesdata[y+np1+1,x]!=spk_num and spokesdata[y+np1+1,x]!=exp_spk and ((datapoints[y+np1+1][x]-brightness)<(thread*brightness)) and (datapoints[y+np1+1][x]-brightness)<brightref:
							spokesdata[y+np1+1,x]=exp_spk
							changenewi[totchange]=y+np1+1
							changenewj[totchange]=x
							totchange=totchange+1
							Totchange_sum=Totchange_sum+1
					
					if (x+np1+1<n) and min([abs(x+np1+1-z) for z in spkj])<0.2*n:
						if spokesdata[y,x+np1+1]!=spk_num and spokesdata[y,x+np1+1]!=exp_spk and ((datapoints[y][x+np1+1]-brightness)<(thread*brightness)) and (datapoints[y][x+np1+1]-brightness)<brightref:
							spokesdata[y,x+np1+1]=exp_spk
							changenewi[totchange]=y
							changenewj[totchange]=x+np1+1
							totchange=totchange+1
							Totchange_sum=Totchange_sum+1
		
					if (y-np1-1>0) and min([abs(y-np1-1-z) for z in spki])<0.5*m:
						if spokesdata[y-np1-1,x]!=spk_num and spokesdata[y-np1-1,x]!=exp_spk and ((datapoints[y-np1-1][x]-brightness)<(thread*brightness)) and (datapoints[y-np1-1][x]-brightness)<brightref:
							spokesdata[y-np1-1,x]=exp_spk
							changenewi[totchange]=y-np1-1
							changenewj[totchange]=x
							totchange=totchange+1
							Totchange_sum=Totchange_sum+1
					
					if (x-np1-1>0) and min([abs(x-np1-1-z) for z in spkj])<0.2*n:
						if spokesdata[y,x-np1-1]!=spk_num and spokesdata[y,x-np1-1]!=exp_spk and ((datapoints[y][x-np1-1]-brightness)<(thread*brightness)) and (datapoints[y][x-np1-1]-brightness)<brightref:
							spokesdata[y,x-np1-1]=exp_spk
							changenewi[totchange]=y
							changenewj[totchange]=x-np1-1
							totchange=totchange+1
							Totchange_sum=Totchange_sum+1
		#print(len(changenewi[changenewi!=-5]))
		#if len(changenewi)>totpix*0.25:
		#	break
		changenewior=changenewi
		changenewjor=changenewj
	return spokesdata
################################################################################################################################################################
################################################################################################################################################################
########################################### end of expanding spokes based on origional spokes ##################################################################
################################################################################################################################################################
################################################################################################################################################################
		

################################################################################################################################################################
################################################################################################################################################################
########################################### start of other useful functions ####################################################################################
################################################################################################################################################################
################################################################################################################################################################
# this function sorts spoke numbers so any empty id number will be eliminated (for example spoke 5 existed but got cleaned out)
def sortspk(spokesdata,spknum_range):
	# spknum_range: spoke id range (need to be sorted)
	idint=0
	spkrealr_end=2*bound
	for i in spknum_range:
		wherei,wherej=np.where(spokesdata==i)
		if len(wherei)!=0:
			spkrealr_end=spkrealr_end+1
			spokesdata[wherei,wherej]=spkrealr_end
	return list(range(2*bound+1,spkrealr_end+1)),spokesdata
	
import collections
# this function cleans the spokes on the edge
def cleanedge_spk(spokesdata,spknum_range,edgeper):
	# spknum_range: spoke id range
	# edgeper: 0.1 means the edge is at 10% the total pixel, get rid of any spokes that are mostly in that reagion 
	m,n=spokesdata.shape
	edgepix=n*edgeper
	for i in spknum_range:
		wherei,wherej=np.where(spokesdata==i)
		counter=collections.Counter(wherei)
		# get rid of right edges, seems to be the only ones that are causing problems
		#print(sum([j>edgepix for j in wherej]))
		#print(len(wherej))
		#print(np.var(counter.values()))
		maxj=max(wherej)
		#print(len(counterj))
		if sum([j>edgepix for j in wherej])>0.8*len(wherej) and (maxj==n-2):
			counterj=len(np.where(wherej==n-2)[0])
			if ((np.var(list(counter.values()))<1200) or (counterj)>0.5*m):
				spokesdata[wherei,wherej]=0
				sizbox=len(np.unique(wherei))*len(np.unique(wherej)) # size of box
				#print((float(sizbox)-float(len(wherei)))/float(sizbox))
				#print('variance',np.var(counter.values()))
	spknum_range,spokesdata=sortspk(spokesdata,spknum_range)
	#print(spknum_range)
	return spknum_range,spokesdata
################################################################################################################################################################
################################################################################################################################################################
########################################### end of other useful functions ######################################################################################
################################################################################################################################################################
################################################################################################################################################################

