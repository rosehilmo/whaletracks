#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:21:00 2021

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

THRESHOLD=[200]
    #Load call dataframes
A_df0=pd.read_csv(AXB_FILE) #load auto_picked calls
A_df=A_df0[['peak_time', 'peak_signal','peak_frequency']].copy()
#b_df.info()
A_df['peak_time']=A_df['peak_time'].apply(lambda x:x[:10])

for threshold in THRESHOLD:
    
    A_df_sub=A_df[A_df['peak_signal']>=threshold]
    A_df_f=A_df_sub.round({'peak_frequency':1})
    f_list=A_df_f['peak_frequency'].values.tolist()
    #df0=pd.DataFrame(A_df_f['peak_frequency'].value_counts(sort=True))
    #df0=df0.reset_index()
    #df0.columns=['peak_frequency','ct']
    #df0=df0.sort_values(by=['peak_frequency'])
    A_df_high=A_df_sub[A_df_sub['peak_frequency']>=20]
    high_list=A_df_high['peak_frequency'].values.tolist()
    A_df_low=A_df_sub[A_df_sub['peak_frequency']<20]
    low_list=A_df_low['peak_frequency'].values.tolist()
#print(freq_cter,len(freq_cter))
#print(type(freq_cter))
#print(df0.head(60))
#get fraction of each type of call
total_ct=len(low_list)+len(high_list)
high_frac=len(high_list)/total_ct*100
low_frac=len(low_list)/total_ct*100
print(high_frac, low_frac)


fig=plt.figure(figsize=(10,7))
plt.hist(f_list, density=True, bins=35)
#plt(,'--')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlabel('Frequency (Hz)',fontsize=20)
plt.xticks(fontsize=20,rotation=45)
plt.ylabel( 'Percentage',fontsize=20,rotation=0,labelpad=60)
plt.yticks(np.linspace(0,0.4,5),fontsize=20,rotation=45)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0, symbol=None))

#add a straight line at freq=20 indicating the boundary of two type of calls
xx=[20,20]
yy=[0,0.5]
plt.plot(xx,yy,'--', color='gray')
#add text indicating high and low freq calls
plt.text(16.5,0.45, 'low frequency\n    49.2%',size=20)
plt.text(21.6,0.45, 'high frequency\n     50.8%',size=20)
plt.show()

'''
data= [high_list, low_list]
ALPHA=0.7
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(111)
ax0.boxplot(data)
plt.legend()
plt.show()
'''