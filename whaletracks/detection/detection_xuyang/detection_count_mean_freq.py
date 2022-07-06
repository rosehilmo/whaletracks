#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 01:30:55 2021

@author: Xuyang
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

AXB_FILE='Fins_chunk_AXBA1_2016_2hr_250.csv' #File name where verified calls will be saved
#HYS_FILE='Fins_chunk_HYSB1_2016_2hr_250.csv' #File name where verified calls will be saved


A_df = pd.read_csv(AXB_FILE, usecols = ['peak_time', 'peak_signal', 'peak_frequency'])
#A_df.set_index('peak_time',inplace = True)
#H_df = pd.read_csv(HYS_FILE, usecols = ['peak_time', 'peak_signal', 'peak_frequency'])

#filter df by threshold
THRESHOLD = 2000

A_df_2000 = A_df[A_df['peak_signal']>=THRESHOLD]
A_df_2000['peak_time']=A_df_2000['peak_time'].apply(lambda x:x[:7])

A_high = A_df_2000[A_df_2000['peak_frequency']>=20].drop(columns='peak_signal')
A_low = A_df_2000[A_df_2000['peak_frequency']<20].drop(columns='peak_signal')

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].set_title('Mean frequency for high frequency calls')
sns.boxplot(ax=axes[1], data = A_low, x='peak_time', y='peak_frequency')
axes[1].set_title('Mean frequency for low frequency calls')
sns.boxplot(ax=axes[0], data = A_high, x='peak_time', y='peak_frequency')



'''
A_high.info()
print(A_low.head())
'''