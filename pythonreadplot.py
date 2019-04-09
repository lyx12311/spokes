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

filename='/Users/lucy/Desktop/spokes/SPKMVLFLP_081/W1597991995_1_cal.rpj1'

# read files and set up initial conditions
a=idlsave.read(filename)
datapoints=a.rrpi # load data in

m,n=datapoints.shape
plt.figure()
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
print('image size: ',(m,n))

a.mnlon,a.mxlon,datapoints=cropdata(datapoints,a.mxlon,a.mnlon)
m,n=datapoints.shape
print('image size: ',(m,n)) # get the shape of the brightness data
totpix=m*n

# longtitude and radian array
lon_array=np.linspace(a.mnlon,a.mxlon,n)
rad_array=np.linspace(a.mnrad,a.mxrad,m)

print('total number of pixcel:, ',totpix)

plt.figure()
plt.imshow(datapoints, cmap = plt.get_cmap('gray'),origin='upper')
plt.show()
