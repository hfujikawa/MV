# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:32:00 2019
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
@author: hfuji
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def average_grad_mag(image):
	gy, gx = np.gradient(image)
	gnorm = np.sqrt(gx**2 + gy**2)
	sharpness = np.average(gnorm)
	return sharpness

target_dir = 'C:/Users/Public/Documents/MVTec/HALCON-18.11-Steady/examples/images/dff/focus_bga'
images_list = glob.glob(target_dir + '*.png')

fmlist = []
gmlist = []
dmlist = []
# loop over the input images
for imagePath in images_list:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	text = "Not Blurry"
 
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < 15.0:
		text = "Blurry"
	
	# https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
	gm = average_grad_mag(gray)
	
	dev = np.std(image)
    
	# show the image
#	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#	cv2.imshow("Image", image)
#	key = cv2.waitKey(0)
	print(text, fm, gm)
	plt.imshow(image)
	plt.show()
    
	fmlist.append(fm)
	gmlist.append(gm)
	dmlist.append(dev*dev)

x = np.arange(0, len(fmlist), 1)
plt.plot(x, fmlist, 'r', x, gmlist, 'b', dmlist, 'g')
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(fmlist, 'r')
ax2 = ax1.twinx()  # 2つのプロットを関連付ける
ax2.plot(dmlist, 'g')
plt.show()

# https://qiita.com/hik0107/items/3dc541158fceb3156ee0
sns.distplot(fmlist, kde=False, rug=False, bins=10) 
sns.distplot(dmlist, kde=False, rug=False, bins=10) 

plt.scatter(fmlist, dmlist)
