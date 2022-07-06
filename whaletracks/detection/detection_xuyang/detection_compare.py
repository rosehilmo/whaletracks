#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:13:36 2021

@author: Xuyang
"""

#This code makes spectrograms and plots fin detection times and scores 
#from the files produced by main_detection for verification 

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read, read_inventory
import pandas as pd
from detect_calls import plotwav
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.colors as color
import sys
import os
sys.path.append('/Users/Xuyang/Documents/github/whaletracks/whaletracks/') 
from common.util import datestrToEpoch

#Define constants
CLIENT_CODE = 'IRIS'
HALF_HOUR = 1800  # in seconds
CHUNK_LENGTH=HALF_HOUR/3   #seconds in spectrogram
PLOTFLAG=True #True makes plots
PLT_SCORE=4 #figure number
#for detection at axial base
#CHUNK_FILE='Fins_chunk_Axial.csv' #File name where verified calls will be saved
#DETECTION_FILE='looking_for_calls_axba1.csv' #File name produced by main_detection


#for detection at hys
#CHUNK_FILE='Fins_chunk_HYSB1_0.5_dist2.csv' #File name where predicted calls will be saved
DETECTION_FILE='looking_for_calls_AXBA1_0.5_dist5.csv' #File name produced by main_detection
MANUAL_PICKED_FILE='Fins_chunk_manual_AXBA1.csv'  #File name produced by manual_picked.py


is_restart='True' #each iteration reads CHUNK_FILE and adds new verified calls
station_id_list=['HYSB1'] #List each station you want to verify
threshold_array=np.linspace(0,2000,20)
threshold_list=threshold_array.tolist()
    #Load call dataframes
b_df=pd.read_csv(DETECTION_FILE) #load auto_picked calls
true_calls=pd.read_csv(MANUAL_PICKED_FILE) #load manully picked calls

    #Either load working verification file or make a new one
verified_calls = pd.DataFrame(columns=b_df.columns)
verified_calls['quality']=[]
chunk_verified=pd.DataFrame(columns=b_df.columns)

false = [] #list of false over total detection ratios
missed = [] #list of missed over true call ratios

for threshold in threshold_list:
    b_df_sub=b_df[b_df['peak_signal']>=threshold]
    b_peak=b_df_sub['peak_epoch'].values.tolist()
    m_peak=true_calls['time'].values.tolist()
    for station_ids in station_id_list:   
    
    #quality_list=[]
        #import pdb; pdb.set_trace()
        true_detect=0
        for inds in range(0,len(m_peak)):
            ind=inds-1
            diff_list=abs(np.subtract(b_peak,m_peak[ind]))
            diff=min(diff_list)
            if diff < 5:
                k=np.argmin(diff_list)
                chunk_verified = chunk_verified.append(b_df_sub.iloc[k])
                #quality_list=quality_list+[1]
                true_detect+=1

        df_false=pd.concat([b_df_sub,chunk_verified]).drop_duplicates(keep=False)
        #quality_list=quality_list+[0]*len(df_false)
        false_detect=len(df_false)
    
        #df_miss=pd.concat([true_calls,chunk_verified]).drop_duplicates(keep=False)
        missed_detect=len(m_peak)-true_detect

        print('Station: {}   Threshold: {}\nTrue detection: {}\nFalse detection: {}\nMissed detection: {}'.format(station_ids,threshold,true_detect,false_detect,missed_detect))
        fal_r = false_detect / (false_detect + true_detect) * 100
        mis_r = missed_detect / (missed_detect + true_detect) * 100
        false.append(fal_r)
        missed.append(mis_r)
        #import pdb; pdb.set_trace()

df=pd.DataFrame(
    {'missed':missed,
     'false':false,
     'threshold':threshold_list
     })

#print(df)
fig = px.scatter(df, x="missed", y="false", text="threshold")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='ROC Curve')
fig.show()




