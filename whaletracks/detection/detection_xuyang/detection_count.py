#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:54:07 2021

@author: wxuyang
"""

#This code makes spectrograms and plots fin detection times and scores 
#from the files produced by main_detection for verification 


from obspy import UTCDateTime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pylab as pylab


#   sys.path.append('/Users/Xuyang/Documents/github/whaletracks/whaletracks/detection/') 

#for detection at axial base
AXB_FILE='Fins_chunk_AXBA1_2016_2hr_250.csv' #File name where verified calls will be saved
HYS_FILE='Fins_chunk_HYSB1_2016_2hr_250.csv' #File name where verified calls will be saved

THRESHOLD=[2000]
    #Load call dataframes
A_df0=pd.read_csv(AXB_FILE) #load auto_picked calls
A_df=A_df0[['peak_time', 'peak_signal']].copy()
#b_df.info()
A_df['peak_time']=A_df['peak_time'].apply(lambda x:x[:10])

H_df0=pd.read_csv(HYS_FILE) #load auto_picked calls
H_df=H_df0[['peak_time', 'peak_signal']].copy()
#b_df.info()
H_df['peak_time']=H_df['peak_time'].apply(lambda x:x[:10])


for threshold in THRESHOLD:
    
    A_df_sub=A_df[A_df['peak_signal']>=threshold]
    A_df_sub=A_df_sub['peak_time'].value_counts(sort=True)
    A_df_f=A_df_sub.reset_index()
    A_df_f.columns=['peak_time','count']
    A_df_f=A_df_f.sort_values(by=['peak_time']).set_index('peak_time')
    
    A_df_f['peak_time'] = pd.to_datetime(A_df_f['peak_time'])
    A_df_f['count'] = pd.to_numeric(A_df_f['count'])
    A_df_f.resample('W').sum()
    
    H_df_sub=H_df[H_df['peak_signal']>=threshold]
    H_df_sub=H_df_sub['peak_time'].value_counts(sort=True)
    H_df_f=H_df_sub.reset_index()
    H_df_f.columns=['peak_time','count']
    H_df_f=H_df_f.sort_values(by=['peak_time']).set_index('peak_time')
    
    H_df_f['peak_time'] = pd.to_datetime(H_df_f['peak_time'])
    H_df_f['count'] = pd.to_numeric(H_df_f['count'])
    H_df_f.resample('W').sum()

ALPHA=0.7
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
'''
ax0.bar(A_df_f.index, A_df_f['count'],color='royalblue')
ax0.title.set_text('Detection count at Station AXBA1')
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax0.grid(alpha=ALPHA)

ax1.bar(H_df_f.index, H_df_f['count'],color='orangered')
ax1.title.set_text('Detection count at Station HYSB1')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.grid(alpha=ALPHA)

plt.ylabel('Number\nof\ndetections',rotation=0,multialignment='center',labelpad=50)
plt.gcf().autofmt_xdate() # Rotation

params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
#plt.legend()
plt.show()
