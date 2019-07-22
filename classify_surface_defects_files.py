# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 05:54:04 2019

@author: hfuji
"""

import os
import glob
import cv2
import matplotlib.pyplot as plt

target_dir = 'D:/Develop/data/Surface-Inspection-defect-detection-dataset/NEU-CLS/'
clsname = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
out_dir = 'D:/Develop/Halcon/TrainDefectImages/'
for clsdir in clsname:
    clsdirpath = out_dir + clsdir
    os.mkdir(clsdirpath)

files = glob.glob(target_dir + '*.bmp')
size = (224, 224)

for fpath in files:
    img = cv2.imread(fpath)
    img_resize = cv2.resize(img, size)
    plt.imshow(img_resize)
    plt.show()
    
    basename = os.path.basename(fpath)
    for clsdir in clsname:
        if clsdir in basename:
            out_path = out_dir + clsdir + '/' + basename[:-4] + '.jpg'
            cv2.imwrite(out_path, img_resize)
    