# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:07:22 2019
https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
@author: hfuji
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

width = 320
height = 256

img = np.zeros((height, width), np.uint8)

for i in range(100):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (200,100)
    fontScale              = 1
    fontColor              = 255
    lineType               = 2
    text                   = str(i)
    
    img_dup = img.copy()
    cv2.putText(img_dup, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    plt.imshow(img_dup, cmap='gray')
    plt.show()
    

#Display the image