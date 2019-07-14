# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 05:48:52 2019

@author: hfuji
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

fpath = 'C:/Users/Public/Documents/MVTec/HALCON-11.0/examples/images/pendulum/pendulum_00.png'
img = cv2.imread(fpath, 0)
height, width = img.shape[:2]

bgnd = np.zeros((height,width), np.uint8)
center = (int(width/2), int(height/2))
radius = 50
cv2.circle(bgnd, center, radius, 255, -1)
result = cv2.bitwise_and(img, bgnd)

plt.imshow(result)
plt.show()