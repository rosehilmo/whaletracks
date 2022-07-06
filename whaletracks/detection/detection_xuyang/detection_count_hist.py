#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 11:18:09 2021

@author: Xuyang
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab



AXB_FILE='Fins_chunk_AXBA1_2016_2hr_250.csv' #File name where verified calls will be saved
HYS_FILE='Fins_chunk_HYSB1_2016_2hr_250.csv' #File name where verified calls will be saved


A_df = pd.read_csv(AXB_FILE, usecols = ['peak_time', 'peak_signal', 'peak_frequency'])
#A_df.set_index('peak_time',inplace = True)
H_df = pd.read_csv(HYS_FILE, usecols = ['peak_time', 'peak_signal', 'peak_frequency'])

#filter df by threshold
THRESHOLD = 2000

A_df_2000 = A_df[A_df['peak_signal']>=THRESHOLD]
A_df_2000['peak_time']=A_df_2000['peak_time'].apply(lambda x:x[:7])
A_df_2000['peak_time'] = pd.to_datetime(A_df_2000['peak_time'])
A_df_2000=A_df_2000.sort_values(by=['peak_time'])

H_df_2000 = H_df[H_df['peak_signal']>=THRESHOLD]
H_df_2000['peak_time']=H_df_2000['peak_time'].apply(lambda x:x[:7])
H_df_2000['peak_time'] = pd.to_datetime(H_df_2000['peak_time'])
H_df_2000=H_df_2000.sort_values(by=['peak_time'])
'''  
H_df_2000 = H_df[H_df['peak_signal']>=THRESHOLD]
H_df_2000['peak_time']=H_df_2000['peak_time'].apply(lambda x:x[:10])
H_df_sub=H_df_2000['peak_time'].value_counts(sort=True)
H_df_f=H_df_sub.reset_index()
H_df_f.columns=['peak_time','count']
H_df_f['peak_time'] = pd.to_datetime(H_df_f['peak_time'])
H_df_f['count'] = pd.to_numeric(H_df_f['count'])
H_df_f=H_df_f.sort_values(by=['peak_time']).set_index('peak_time')
'''
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.hist(A_df_2000['peak_time'], bins=12, density=True)
ax1.hist(H_df_2000['peak_time'], bins=12, density=True)




'''
A_resample = A_df_2000.resample('w').agg({'peak_signal':'mean', 'peak_frequency':'max'})
peak_freq = A_df_2000['peak_signal'].resample('M').max() #only works with start and end frequency. 
peak_freq.plot()
'''
'''
A_df_f.info()
#A_resample.info()
print(A_df_f.head())
'''