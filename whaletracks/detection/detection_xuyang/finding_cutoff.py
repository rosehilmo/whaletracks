#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 03:33:58 2021

@author: Xuyang
"""

from obspy import UTCDateTime
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
#   sys.path.append('/Users/Xuyang/Documents/github/whaletracks/whaletracks/detection/') 

#for detection at axial base
AXB_FILE='Fins_chunk_AXBA1_2016_2hr_250.csv' #File name where verified calls will be saved
#AXB2_FILE=''
HYS_FILE='Fins_chunk_HYSB1_2016_2hr_250.csv' #File name where verified calls will be saved


A_df = pd.read_csv(AXB_FILE, usecols = ['peak_signal', 'peak_frequency'])

H_df = pd.read_csv(HYS_FILE, usecols = ['peak_signal', 'peak_frequency'])

THRESHOLD=350
    #Load call dataframes
#b_df.info()
A_df_sub = A_df[A_df['peak_signal']>=THRESHOLD]
A_df_f=A_df_sub.round({'peak_frequency':1})
A_f=pd.DataFrame(A_df_f['peak_frequency'].value_counts(sort=True))
A_f=A_f.reset_index()
A_f.columns=['peak_frequency','ct']
A_f=A_f.sort_values(by=['peak_frequency'])

H_df_sub = H_df[H_df['peak_signal']>=THRESHOLD]
H_df_f=H_df_sub.round({'peak_frequency':1})
H_f=pd.DataFrame(H_df_f['peak_frequency'].value_counts(sort=True))
H_f=H_f.reset_index()
H_f.columns=['peak_frequency','ct']
H_f=H_f.sort_values(by=['peak_frequency'])

fig, ((ax0,ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15,10))
ax0.bar(x=A_f['peak_frequency'], height=A_f['ct'])
ax1.bar(x=A_f['peak_frequency'], height=A_f['ct'])
ax2.bar(x=H_f['peak_frequency'], height=H_f['ct'])
    