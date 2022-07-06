#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 21:48:17 2021

@author: Xuyang
"""

import sys
sys.path.append('/Users/xuyang/documents/GitHub/whaletracks/whaletracks') 
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import math
import numpy.matlib
import csv
from scipy.signal import find_peaks
import statistics
import pandas as pd


#import the slope list file as a dataframe
#df = pd.read_csv('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_list.csv', sep=',', header=Non)
slope_list1 = []
slope_list2 = []

#create the list of high frequency calls in 2015
with open('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_list_2015_high.csv') as temp_f:
    lines = temp_f.readlines()
    for l in lines:
        slope_string = l.split(',')
        for string in slope_string:
            slope_temp = string.split(',')
            slope_val = [s.replace("\n", "0") for s in slope_temp]
            #print(slope_val)
            if int(float(slope_val[0])) < -10 and int(float(slope_val[0])) > -20:
                slope_list1.append(float(slope_val[0]))
#append the list of low frequency calls in 2015 to previous
with open('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_list_2015_low.csv') as temp_f:
    lines = temp_f.readlines()
    for l in lines:
        slope_string = l.split(',')
        for string in slope_string:
            slope_temp = string.split(',')
            slope_val = [s.replace("\n", "0") for s in slope_temp]
            #print(slope_val)
            if int(float(slope_val[0])) < -10 and int(float(slope_val[0])) > -20:
                slope_list1.append(float(slope_val[0]))
#create the list of high frequency calls in 2015
with open('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_list_2020_high.csv') as temp_f:
    lines = temp_f.readlines()
    for l in lines:
        slope_string = l.split(',')
        for string in slope_string:
            slope_temp = string.split(',')
            slope_val = [s.replace("\n", "0") for s in slope_temp]
            #print(slope_val)
            if int(float(slope_val[0])) < -10 and int(float(slope_val[0])) > -20:
                slope_list2.append(float(slope_val[0]))
#create the list of high frequency calls in 2015
with open('/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_list_2020_low.csv') as temp_f:
    lines = temp_f.readlines()
    for l in lines:
        slope_string = l.split(',')
        for string in slope_string:
            slope_temp = string.split(',')
            slope_val = [s.replace("\n", "0") for s in slope_temp]
            #print(slope_val)
            if int(float(slope_val[0])) < -10 and int(float(slope_val[0])) > -20:
                slope_list2.append(float(slope_val[0]))


slope_dict = {'2015':slope_list1, '2020':slope_list2}

fig, ax = plt.subplots()
ax.boxplot(slope_dict.values())
ax.set_xticklabels(slope_dict.keys())
plt.show()
#print(slope_list1)
#print(len(slope_list1), len(slope_list2))
print("During fall and winter seasons of 2015\nThe mean of slopes:\n{} \nThe standard deviation:\n{}]".format(statistics.mean(slope_list1), statistics.stdev(slope_list1)))
print("During fall and winter seasons of 2020\nThe mean of slopes:\n{} \nThe standard deviation:\n{}]".format(statistics.mean(slope_list2), statistics.stdev(slope_list2)))


