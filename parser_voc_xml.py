# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:27:10 2019
https://qiita.com/haru1977/items/d30e9730abfa9a193285
@author: hfuji
"""

import xml.etree.ElementTree as ET
import os
import glob
import cv2
import matplotlib.pyplot as plt

src_dir = 'C:/Users/Public/Documents/MVTec/HALCON-18.11-Steady/examples/images/pill_bag/'
dst_dir = 'D:/Develop/data/VOCdevkitCOCO/VOCCOCO/JPEGImages/'
FILE = 'D:/Develop/data/VOCdevkitCOCO/VOCCOCO/Annotations/000000000001.xml'

files = glob.glob(os.path.dirname(FILE) + '/*.xml')

for fpath in files:
#    file = open(FILE)
    file = open(fpath)
    tree = ET.parse(file)
    root = tree.getroot()
    
    all_list = []
    
    img_file = root.find('filename').text  # 画像ファイル名を取得
    
    img_path = src_dir + img_file
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.show()
    basename = os.path.basename(fpath)
    dst_path = dst_dir + basename[:-4] + '.jpg'
    cv2.imwrite(dst_path, img)
    
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')