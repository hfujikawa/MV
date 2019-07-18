# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 05:46:25 2019
https://stackoverflow.com/questions/33523751/getting-specific-frames-from-videocapture-opencv-in-python
@author: hfuji
"""

import cv2

avi_path = 'D:/test.avi'
cap = cv2.VideoCapture(avi_path)
total_frames = cap.get(7)

out_path = 'D:/frame100.jpg'
cap.set(1, 100)
ret, frame = cap.read()
cv2.imwrite(out_path, frame)
