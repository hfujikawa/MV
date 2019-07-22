# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 08:58:39 2019

@author: hfuji
"""

import os
import glob
import shutil

target_dir = 'D:/Develop/data/raccoon_dataset/'
png_files = glob.glob(target_dir + 'PngImages/*.png')
xml_files = glob.glob(target_dir + 'Annotations/*.xml')

for fpath in xml_files:
    basename = os.path.basename(fpath)
    fname = basename[:-4]
    words = fname.split('-')
    id_num = '{0:03d}'.format(int(words[1]))
    dstname = words[0] + '-' + id_num + '.xmll'
    dirname= os.path.dirname(fpath)
    dstpath = dirname + '/' + dstname
    shutil.move(fpath, dstpath)