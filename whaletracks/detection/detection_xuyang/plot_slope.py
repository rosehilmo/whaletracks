#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:40:28 2021

@author: Xuyang
"""

import numpy as np
import matplotlib.pyplot as plt
t_series, freq_series = np.genfromtxt(fname='/Users/xuyang/documents/GitHub/whaletracks/whaletracks/detection/slope_coor.csv',skip_header=4,delimiter=',',unpack=True)
#print(len(t_series))
t_start = []
t_end = []
freq_start = []
freq_end = []
for i in range(0,len(t_series)):
    if i%2 == 0:
        t_start.append(t_series[i])
        freq_start.append(freq_series[i])
    else:
        t_end.append(t_series[i])
        freq_end.append(freq_series[i])
#print(t_start, t_end) 
t_int = []
freq_int = []
for i in range(0, len(t_start)):
    t_int.append(t_end[i] - t_start[i])
    freq_int.append(freq_start[i] - freq_end[i])
print(t_int)
plt.plot(t_int, freq_int, 'ro')
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')
plt.show()