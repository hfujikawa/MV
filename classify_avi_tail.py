# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:36:56 2019

@author: hfuji
"""

import numpy as np
import cv2
#import glob
#import time
import matplotlib.pyplot as plt
import seaborn as sns

video_path0 = 'C:/Users/hfuji/Videos/admt201800136-sup-0001-s1.mov'
video_path1 = 'C:/Users/hfuji/Videos/admt201800136-sup-0002-s2.mov'
video_path2 = 'C:/Users/hfuji/Videos/admt201800136-sup-0004-s4.mov'
video_path_grp = [video_path0, video_path1, video_path2]

rect_w_list_grp = []
for idx, video_path in enumerate(video_path_grp):
    cap = cv2.VideoCapture(video_path)
    
    frame_cnt = 0
    rect_w_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True:
            break
        height, width = frame.shape[:2]
        bgnd = np.zeros((width, height), np.uint8)
        
        cv2.imshow('frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        area = cv2.countNonZero(bin_img)
        if area < 1:
            continue
        
        _, cnts, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_max = []
        len_max = 0
        for contour in cnts:
            if len(contour) > len_max:
                len_max = len(contour)
                cnts_max = contour
        # https://www.programcreek.com/python/example/89409/cv2.fitEllipse
        ellipse = cv2.fitEllipse(cnts_max)
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        cv2.fillPoly(bgnd, [poly], 255)
        gray = cv2.drawContours(gray, [poly], -1, 255, 2)
        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        rect = cv2.minAreaRect(cnts_max)
        rect_w = rect[1][0]
        rect_h = rect[1][1]
        rect_w_list.append(rect_w)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        gray = cv2.drawContours(gray, [box], 0, 255, 2)
        print(frame_cnt, area)
        plt.imshow(gray, cmap='gray')
        plt.show()
        
    #    if frame_cnt > 10:
    #        break
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #
    #    else:
    #        break
        frame_cnt += 1
    
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    
#    sns.distplot(rect_w_list, kde=False, rug=False, bins=10)
    rect_w_list_grp.append(rect_w_list)

# https://qiita.com/sotetsuk/items/80c57896735aee4b0306
min0 = min(rect_w_list_grp[0])
max0 = max(rect_w_list_grp[0])
range0 = max0 - min0
min1 = min(rect_w_list_grp[1])
max1 = max(rect_w_list_grp[1])
range1 = max1 - min1
min2 = min(rect_w_list_grp[2])
max2 = max(rect_w_list_grp[2])
range2 = max2 - min2
step_min = min(range0, range1, range2) / 10
bins0 = int(range0 / step_min)
bins1 = int(range1 / step_min)
bins2 = int(range2 / step_min)

plt.hist(rect_w_list_grp[0], bins=bins0, alpha=0.3, histtype='stepfilled', color='r')
plt.hist(rect_w_list_grp[1], bins=bins1, alpha=0.3, histtype='stepfilled', color='b')
plt.hist(rect_w_list_grp[2], bins=bins2, alpha=0.3, histtype='stepfilled', color='g')
plt.show()