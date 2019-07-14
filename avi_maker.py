# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:57:10 2019

@author: hfuji
"""

import glob
import cv2
import matplotlib.pyplot as plt

# https://playwithopencv.blogspot.com/2012/07/python-opencv-tutorial4.html
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

# http://pynote.hatenablog.com/entry/opencv-video-capture-and-writer
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20
width = 320
height = 240
#width = 640
#height = 480
writer = cv2.VideoWriter('output2.avi', fourcc, fps, (width, height), True)

src_dir = 'C:/Users/Public/Documents/MVTec/HALCON-11.0/examples/images/'
files = glob.glob(src_dir + 'pendulum/*.png')
#files = glob.glob(src_dir + 'color/*.png')

for fpath in files:
    img = cv2.imread(fpath, 1)
#    result, encimg=cv2.imencode('.jpg', img, encode_param)
    encimg = cv2.imencode('.jpg', img)[1]
#    if False==result:  
#        print('could not encode image!')
#        break
    writer.write(img)
#    writer.write(encimg)
    #decode from jpeg format  
    decimg = cv2.imdecode(encimg, 1)  
    plt.imshow(decimg)
    plt.show()

writer.release()

"""
device_id = 0
cap = cv2.VideoCapture(device_id)
cap.set(cv2.CAP_PROP_FPS, fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break  # 映像取得に失敗
    result, encimg=cv2.imencode('.jpg', frame, encode_param)  
    if False==result:  
        print('could not encode image!')
        break
#    writer.write(encimg)
    writer.write(frame)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # q キーを押したら終了する。

cap.release()
cv2.destroyAllWindows()
"""