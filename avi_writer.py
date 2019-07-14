# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:49:48 2019
http://pynote.hatenablog.com/entry/opencv-video-capture-and-writer
@author: hfuji
"""

import numpy as np
import cv2
import glob
import time

cap = cv2.VideoCapture(0)
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_divx.avi',fourcc, 20.0, (640,480), False)

src_dir = 'C:/Users/Public/Documents/MVTec/HALCON-11.0/examples/images/'
files = glob.glob(src_dir + 'color/*.png')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
#        frame = cv2.flip(frame,0)
        result, encimg=cv2.imencode('.jpg', frame, encode_param)  
        if False==result:  
            print('could not encode image!')
            break
        start = time.time()
        # write the flipped frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
#        out.write(encimg)S
        end = time.time()
        print(end - start)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()