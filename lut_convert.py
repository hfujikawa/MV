# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:24:03 2019
https://www.ncl.ucar.edu/Document/Graphics/ColorTables/matlab_jet.shtml
https://qiita.com/como1559/items/abfcd4f0ce2f611d5193
@author: hfuji
"""
import re

lut_table = []
with open("D:/Develop/Halcon/matlab_jet.rgb", "rt") as fp:
    lines = fp.readlines()
    for idx, line in enumerate(lines):
        if idx < 2:
            continue
        line = line.strip()
        words = re.split(' +', line)
        lut_table.append(words[0] + ' ' + words[1] + ' ' + words[2])

with open("jet.lut", "wt") as fpout:
    fpout.writelines('\n'.join(lut_table))