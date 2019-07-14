# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:35:26 2019
https://stackoverflow.com/questions/45267727/python-opcencv-split-recording-into-multiple-files
@author: hfuji
"""

import numpy as np
import cv2
import time


def get_output(out=None):
    #Specify the path and name of the video file as well as the encoding, fps and resolution
    if out:
        out.release()
    return cv2.VideoWriter('D:/' + str(time.strftime('%d %m %Y - %H %M %S' )) + '.avi', cv2.cv.CV_FOURCC('X','V','I','D'), 15, (640,480))

cap = cv2.VideoCapture(0)
next_time = time.time() + 10
out = get_output()

while True:
    if time.time() > next_time:
        next_time += 10
        out = get_output(out)

    # Capture frame-by-frame
    ret, frame = cap.read() 

    if ret:
        out.write(frame)
    cv2.imshow('camera capture', frame)
    k = cv2.waitKey(1) # 1msec待つ
    if k == 27: # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()