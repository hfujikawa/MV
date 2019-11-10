# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 07:51:23 2019

@author: hfuji
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

target_dir = 'C:/Users/Public/Documents/MVTec/HALCON-18.11-Steady/examples/images/dff/focus_bga'
images_list = glob.glob(target_dir + '*.png')

diam_list = []     # diameter of melt pool
gvar_list = []     # gray variance
xdat_list = []

# loop over the input images
center = (0, 0)
radius = 0
xs = 426
ys = 407
xw = 314
yh = 262
roi_size = int(np.sqrt(47982*2))
for idx, imagePath in enumerate(images_list):
    if idx < 5 or idx > 12:
        continue
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # measure the gray variance
    roi = gray[ys:ys+roi_size, xs:xs+roi_size]
    gvar_list.append(np.std(roi)**2)
    
    # measure the diameter of melt pool
    height, width = gray.shape[:2]
  # 二値化
    ret, img_bin = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    area = cv2.countNonZero(img_bin)
    plt.imshow(img_bin, cmap='gray')
    plt.show()
    
    # 輪郭抽出
    _, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_max = []
    len_max = 0
    for contour in contours:
        if len(contour) > len_max:
            len_max = len(contour)
            cnts_max = contour
    # https://www.programcreek.com/python/example/89409/cv2.fitEllipse
    ellipse = cv2.fitEllipse(cnts_max)
    diam_list.append(int(ellipse[1][0]))
    xdat_list.append(idx)
    poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    if idx == 8:
        x, y, w, h = cv2.boundingRect(cnts_max)
        bgnd = np.zeros((height, width), np.uint8)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        radius = int(ellipse[1][0])
#        cv2.fillPoly(bgnd, [poly], 255)
        gray = cv2.drawContours(gray, [poly], -1, 255, 2)
        bgnd = cv2.circle(bgnd, center, radius, 255, -1)
        plt.imshow(gray, cmap='gray')
        plt.show()
        plt.imshow(bgnd, cmap='gray')
        plt.show()
#        break
    
#    if idx > 2:
#        break
fig, ax1 = plt.subplots()
ax1.plot(xdat_list, gvar_list, color='b')
ax2 = ax1.twinx()
ax2.plot(xdat_list, diam_list, color='r')
plt.show()

plt.scatter(diam_list, gvar_list)
plt.show()