#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:50:44 2021

@author: Xuyang
"""

import sys
sys.path.append('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection') 
import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics
import pandas as pd
import os


missed = 0
true_ct = 0
false_ct = 0
Station_id = 'HYSB1'

#import the detection file
#for verification at Axial base
#CHUNK_FILE='Fins_chunk_Axial.csv' 
#dur,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,tf=np.genfromtxt(fname='/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/Fins_chunk_Axial.csv',skip_header=1567,delimiter=',',unpack=True)

#for verification at HYSB
CHUNK_FILE='Fins_chunk_Hys.csv' 
dur,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,tf=np.genfromtxt(fname='/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/Fins_chunk_Hys.csv',skip_header=1,delimiter=',',unpack=True)


for i in range(0, len(dur)):
    if np.isnan(dur[i]) == True:
        missed +=1
    else:
        if tf[i] >0.5:
            true_ct +=1
        else:
            false_ct +=1
            
total_ct = missed + true_ct + false_ct           
print('{}@P100\ntrue detections:{}, false detections:{}, missed calls:{}; '.format(Station_id, true_ct, false_ct, missed))
print(true_ct/total_ct, false_ct/total_ct, missed/total_ct)