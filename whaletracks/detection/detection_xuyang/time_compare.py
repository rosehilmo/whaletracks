#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:17:59 2021

@author: Xuyang
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read, read_inventory
import pandas as pd
from detect_calls import plotwav
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import sys
import os
sys.path.append('/Users/Xuyang/Documents/github/whaletracks/whaletracks/') 
from common.util import datestrToEpoch
import csv
import detect_calls as detect
import math


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
CHUNK_FILE='Fins_chunk_HYSB1_0.5_dist5.csv'
DETECTION_FILE='looking_for_calls_HYSB1_0.5_dist5.csv' #File name produced by main_detection
MANUAL_PICKED_FILE='Fins_chunk_manual_HYSB1_0.5_high_qual.csv'  #File name produced by manual_picked.py

F0 = 27 #average start frequency
F1 = 15 #average end frequency
BDWDTH = 4 # average bandwidth
DUR = 0.7#average duration

is_restart='True' #each iteration reads CHUNK_FILE and adds new verified calls
station_id_list=['HYSB1'] #List each station you want to verify
#threshold_list=[0.1,1,5,10,20,50,100,200,500,750,1000,1500,3000,5000,8000]
threshold = 0.5
    #Load call dataframes
b_df=pd.read_csv(DETECTION_FILE) #load auto_picked calls
true_calls=pd.read_csv(MANUAL_PICKED_FILE) #load manully picked calls

#Starttime_list = ["2020-11-15T22:00:00.000"] #high quality for testing
#create a list of time of detection
#For axial base
'''Starttime_list = ["2016-01-04T01:00:00.000",
                  "2016-01-04T01:10:00.000",
                  "2016-01-04T01:20:00.000",
                  "2016-01-04T01:30:00.000",
                  "2020-10-08T04:00:00.000",
                  "2020-10-08T04:10:00.000",
                  "2020-10-08T04:20:00.000",
                  "2020-12-26T02:00:00.000",
                  "2020-12-26T02:10:00.000",
                  "2020-12-26T02:20:00.000",
                  ]
'''

#for hysb1
#save for later
"""
"2015-10-06T02:00:00.000",
                  "2015-10-06T02:10:00.000",
                  "2015-10-06T02:20:00.000",
                  "2015-10-06T02:30:00.000",
                  
                  "2015-11-26T04:00:00.000",
                  
                  "2016-01-01T04:20:00.000",
                  "2016-01-01T04:30:00.000",
                  "2016-01-01T04:40:00.000",
                  "2016-01-01T04:50:00.000",
                  
                  "2016-06-19T21:30:00.000",
                  
                  "2016-11-20T15:10:00.000",
                  
                  "2017-01-01T06:20:00.000",
                  "2017-01-01T06:30:00.000",
                  "2017-01-01T06:40:00.000",
                  
                  "2017-12-05T06:00:00.000",
                  "2017-12-05T06:10:00.000",
                  "2017-12-05T06:20:00.000",
                  
                  "2019-11-23T04:50:00.000",
                  "2020-10-24T02:00:00.000",
                  "2020-10-24T02:10:00.000",
                  "2020-10-24T02:20:00.000",
                  "2020-11-15T22:10:00.000",
                  "2020-11-15T22:20:00.000",
                  "2020-11-15T22:30:00.000",
                  
"""
'''            
Starttime_list = ["2015-09-10T02:10:00.000",
                  "2015-09-10T02:20:00.000",
                  "2015-09-10T02:30:00.000",
                  "2015-09-10T02:40:00.000",
                  "2015-11-26T03:30:00.000",
                  "2015-11-26T03:40:00.000",
                  "2015-11-26T03:50:00.000",
                  "2016-05-01T19:30:00.000",
                  "2016-05-01T19:40:00.000",
                  "2016-05-01T19:50:00.000",
                  "2016-06-19T21:40:00.000",
                  "2016-06-19T21:50:00.000",
                  "2016-06-19T22:00:00.000",
                  "2016-06-19T22:10:00.000",
                  "2016-06-19T22:20:00.000",
                  "2016-06-19T22:30:00.000",
                  "2016-06-19T22:50:00.000",
                  "2016-08-01T04:20:00.000",
                  "2016-08-01T04:30:00.000",
                  "2016-08-01T04:40:00.000",
                  "2016-09-02T05:20:00.000",
                  "2016-09-02T05:30:00.000",
                  "2016-09-02T05:40:00.000",
                  "2016-11-20T15:20:00.000",
                  "2017-04-05T06:20:00.000",
                  "2017-04-05T06:30:00.000",
                  "2017-04-05T06:40:00.000",
                  "2017-07-05T06:20:00.000",
                  "2017-07-05T06:30:00.000",
                  "2017-07-05T06:40:00.000",
                  "2017-10-05T06:00:00.000",
                  "2017-10-05T06:10:00.000",
                  "2017-10-05T06:20:00.000",
                  "2017-11-05T06:00:00.000",
                  "2017-11-05T06:10:00.000",
                  "2017-11-05T06:20:00.000",
                  "2018-04-05T06:20:00.000",
                  "2018-04-05T06:30:00.000",
                  "2018-04-05T06:40:00.000",
                  "2018-07-05T06:20:00.000",
                  "2018-07-05T06:30:00.000",
                  "2018-07-05T06:40:00.000",
                  "2019-04-05T06:20:00.000",
                  "2019-04-05T06:30:00.000",
                  "2019-04-05T06:40:00.000",
                  "2019-07-05T06:20:00.000",
                  "2019-07-05T06:30:00.000",
                  "2019-07-05T06:40:00.000",
                  "2019-10-15T07:00:00.000",
                  "2019-10-15T07:10:00.000",
                  "2019-10-15T07:20:00.000",
                  "2019-10-15T07:30:00.000",
                  ]
'''
'''
#for high quality at HYSB1
Starttime_list = ["2015-10-06T02:00:00.000",
                  "2015-11-26T03:30:00.000",
                  "2015-11-26T03:40:00.000",
                  "2015-11-26T03:50:00.000",
                  "2016-11-20T15:20:00.000",
                  "2019-10-15T07:00:00.000",
                  "2019-10-15T07:10:00.000",
                  "2019-10-15T07:20:00.000",
                  "2019-10-15T07:30:00.000",
                  "2020-11-15T22:00:00.000",
                  "2020-11-15T22:10:00.000",
                  "2020-11-15T22:20:00.000",
                  ]
'''
'''
Starttime_list = ["2015-09-10T02:10:00.000",
                  "2015-09-10T02:20:00.000",
                  "2015-09-10T02:30:00.000",
                  "2015-09-10T02:40:00.000",
                  "2015-10-06T02:00:00.000",
                  "2015-10-06T02:10:00.000",
                  "2015-10-06T02:20:00.000",
                  "2015-10-06T02:30:00.000",
                  "2015-11-26T03:30:00.000",
                  "2015-11-26T03:40:00.000",
                  "2015-11-26T03:50:00.000",
                  "2015-11-26T04:00:00.000",
                  "2016-01-01T04:20:00.000",
                  "2016-01-01T04:30:00.000",
                  "2016-01-01T04:40:00.000",
                  "2016-01-01T04:50:00.000",
                  "2016-05-01T19:30:00.000",
                  "2016-05-01T19:40:00.000",
                  "2016-05-01T19:50:00.000",
                  "2016-06-19T21:30:00.000",
                  "2016-06-19T21:40:00.000",
                  "2016-06-19T21:50:00.000",
                  "2016-06-19T22:00:00.000",
                  "2016-06-19T22:10:00.000",
                  "2016-06-19T22:20:00.000",
                  "2016-06-19T22:30:00.000",
                  "2016-06-19T22:50:00.000",
                  "2016-08-01T04:20:00.000",
                  "2016-08-01T04:30:00.000",
                  "2016-08-01T04:40:00.000",
                  "2016-09-02T05:20:00.000",
                  "2016-09-02T05:30:00.000",
                  "2016-09-02T05:40:00.000",
                  "2016-11-20T15:10:00.000",
                  "2016-11-20T15:20:00.000",
                  "2017-01-01T06:20:00.000",
                  "2017-01-01T06:30:00.000",
                  "2017-01-01T06:40:00.000",
                  "2017-04-05T06:20:00.000",
                  "2017-04-05T06:30:00.000",
                  "2017-04-05T06:40:00.000",
                  "2017-07-05T06:20:00.000",
                  "2017-07-05T06:30:00.000",
                  "2017-07-05T06:40:00.000",
                  "2017-10-05T06:00:00.000",
                  "2017-10-05T06:10:00.000",
                  "2017-10-05T06:20:00.000",
                  "2017-11-05T06:00:00.000",
                  "2017-11-05T06:10:00.000",
                  "2017-11-05T06:20:00.000",
                  "2017-12-05T06:00:00.000",
                  "2017-12-05T06:10:00.000",
                  "2017-12-05T06:20:00.000",
                  "2018-04-05T06:20:00.000",
                  "2018-04-05T06:30:00.000",
                  "2018-04-05T06:40:00.000",
                  "2018-07-05T06:20:00.000",
                  "2018-07-05T06:30:00.000",
                  "2018-07-05T06:40:00.000",
                  "2019-04-05T06:20:00.000",
                  "2019-04-05T06:30:00.000",
                  "2019-04-05T06:40:00.000",
                  "2019-07-05T06:20:00.000",
                  "2019-07-05T06:30:00.000",
                  "2019-07-05T06:40:00.000",
                  "2019-10-15T07:00:00.000",
                  "2019-10-15T07:10:00.000",
                  "2019-10-15T07:20:00.000",
                  "2019-10-15T07:30:00.000",
                  "2019-11-23T04:50:00.000",
                  "2020-10-24T02:00:00.000",
                  "2020-10-24T02:10:00.000",
                  "2020-10-24T02:20:00.000",
                  "2020-11-15T22:10:00.000",
                  "2020-11-15T22:20:00.000",
                  "2020-11-15T22:30:00.000",
                  ]
'''

Starttime_list = ["2015-10-06T02:00:00.000",
                  "2015-11-26T03:30:00.000",
                  "2015-11-26T03:40:00.000",
                  "2015-11-26T03:50:00.000",
                  "2016-01-01T04:20:00.000",
                  "2016-01-01T04:30:00.000",
                  "2016-01-01T04:40:00.000",
                  "2016-01-01T04:50:00.000",
                  "2016-11-20T15:20:00.000",
                  "2019-10-15T07:00:00.000",
                  "2019-10-15T07:10:00.000",
                  "2019-10-15T07:20:00.000",
                  "2019-10-15T07:30:00.000",
                  ]


#Starttime_list = ["2015-10-06T02:00:00.000"]
for time_ind in range(0,len(Starttime_list)):
    STARTTIME = Starttime_list[time_ind] #start time

    STARTEPOCH = datestrToEpoch([STARTTIME],dateformat='%Y-%m-%dT%H:%M:%S.%f')
    ENDEPOCH = [STARTEPOCH[0] + CHUNK_LENGTH]

    utcstart_chunk = UTCDateTime(STARTTIME)
    utcend_chunk = UTCDateTime(STARTTIME) + CHUNK_LENGTH

    client=Client('IRIS')

    #Load call dataframes
    b_df_filter=b_df[b_df['peak_signal']>=threshold]
    
    print(STARTTIME)
    print('Threshold: {}'.format(threshold))
    #Either load working verification file or make a new one
    if os.path.isfile(CHUNK_FILE) and is_restart:
        verified_calls = pd.read_csv(CHUNK_FILE)
    else:
        verified_calls = pd.DataFrame(columns=b_df_filter.columns)
        verified_calls['quality']=[]

    for station_ids in station_id_list:   
        st_raw_exist = False
        retry=0
        chunk_verified=pd.DataFrame(columns=b_df_filter.columns)
        
        #Filter b, a and b filtered dataframes for specific station and date range
        b_df_sub=b_df_filter.loc[(b_df_filter['peak_epoch'] > STARTEPOCH[0]) &
                                 (b_df_filter['peak_epoch'] < ENDEPOCH[0])]
        m_df_sub=true_calls.loc[(true_calls['time'] > STARTEPOCH[0]) &
                                (true_calls['time'] < ENDEPOCH[0])]
        #import pdb; pdb.set_trace()
        #Get waveform from IRIS
        while st_raw_exist == False and retry < 5:
            try:
                st_raw=client.get_waveforms(network="OO", station=station_ids, location='*',
                                            channel='HHZ', starttime=utcstart_chunk,
                                            endtime=utcend_chunk, attach_response=True)
                st_raw_exist=True
                retry=5
            except:
                retry=retry+1
                st_raw_exist=False
                print("Client failed: Retry " + str(retry) + " of 5 attempts")
                
        if st_raw_exist == False:
            print("WARNING: no data available from input station/times")
            continue

        #Filter waveform, remove response, remove sensitivity
        st_raw.detrend(type="demean")
        st_raw.detrend(type="linear")
        st_raw.remove_response(output='VEL',pre_filt=[.2,.5,28,30])
        st_raw.remove_sensitivity()
        
        tr=st_raw[0]

        #specifically select waveform            
        tr_filt=tr.copy()

        #Make spectrogram of data
        window_size=0.8
            
        overlap=.95
        freqlim=[3, 37]
        #SNR metrics
        snr_limits=[15, 25]
        snr_calllength=1
        snr_freqwidth=5
        #Event metrics
        prominence=0.5 #min threshold
        event_dur= .1 #minimum width of detection
        distance=5 #minimum distance between detections
        rel_height=.8
        
        
        [f,t,Sxx]=detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=window_size, overlap=overlap, plotflag=PLOTFLAG,filt_freqlim=freqlim,ylim=freqlim)
            #import pdb; pdb.set_trace()
            #Make detection kernel
        [tvec, fvec, BlueKernel, freq_inds]=detect.buildkernel(F0, F1, BDWDTH, DUR, f, t, tr_filt.stats.sampling_rate, plotflag=PLOTFLAG, kernel_lims=detect.defaultKernelLims)
            
            #subset spectrogram to be in same frequency range as kernel
        Sxx_log=Sxx
#Cross-correlate kernel with spectrogram
        ind1 = 0
        CorrVal = np.zeros(np.size(t) - (len(tvec)-1)) #preallocate array for correlation values
        corrchunk= np.zeros((np.size(fvec), np.size(tvec))) #preallocate array for element-wise multiplication

        while ind1-1+np.size(tvec) < np.size(t):
            ind2 = ind1 + np.size(tvec) #indices of spectrogram subset to multiply
            for indF in range(np.size(fvec)-1):
                corrchunk[indF] = Sxx[indF][ind1:ind2] #grab spectrogram subset for multiplication
        
            CorrVal[ind1] = np.sum(np.multiply(BlueKernel, corrchunk)) #save cross-correlation value for each frame
            ind1 += 1
    
    
        CorrVal_scale=CorrVal*1/(np.median(Sxx_log)*np.size(tvec))
        #CorrVal_scale=CorrVal*1/(np.median(Sxx))
       #CorrVal_scale=CorrVal
        CorrVal_scale[0]=0
        CorrVal_scale[-1]=0
        neg_ind=CorrVal_scale<0
        CorrVal_scale[neg_ind]=0
        t_scale=t[int(len(tvec)/2)-1:-math.ceil(len(tvec)/2)]

        b_peak0=b_df_sub['peak_epoch'].values.tolist()
        b_peak=[i-STARTEPOCH[0] for i in b_peak0]
        b_freq=b_df_sub['peak_frequency'].values.tolist()
        b_score=b_df_sub['peak_signal'].values.tolist()
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        CorrVal_idx=[find_nearest(t_scale,value) for value in b_peak]
        b_scale = [CorrVal_scale[i] for i in CorrVal_idx]
        m_peak=m_df_sub['time'].values.tolist()
        m_freq=m_df_sub['frequency'].values.tolist()
        #import pdb; pdb.set_trace()
        
    if PLOTFLAG==True:
                
            t1=min(t)
            t2=max(t)
            #plot a and b calls on upper axis
            #plt.figure(PLT_SCORE, figsize=(9, 3))
            fig, (ax0,ax1) = plt.subplots(nrows=2,sharex=True)
            
            ax0.plot(t_scale,CorrVal_scale) #plot normalized detection scores as a time series.
            ax0.plot(b_peak,b_scale,'rx')
            ax0.set_xlim([t1, t2]) #look at only positive values
            ax0.set_ylim([0, np.max(CorrVal_scale)])
            ax0.set_xlabel('Seconds')
            ax0.set_ylabel('Detection score')
            ax0.set_title('Detection score, peak time, and width')

        #plot spectrogram 
            cmap = plt.get_cmap('magma')
            vmin=np.median(Sxx_log)+2*np.std(Sxx_log)
            vmax=np.median(Sxx_log)
            norm = color.Normalize(vmin=vmin, vmax=vmax)
            #plt.subplot(212)
            im = ax1.pcolormesh(t, f, Sxx_log, cmap=cmap,norm=norm) 
            fig.colorbar(im, ax=ax1,orientation='horizontal')
            ax1.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
            ax1.set_ylim([12, 32])
            ax1.set_ylabel('Frequency [Hz]')
            #ax1.set_xticks([])
            #ax1.set_xlabel('Time [seconds]')
            fig.tight_layout()
            
            #plot peak frequencies & time
            ax1.plot(b_peak,b_freq,'yx', zorder=1)
            ax1.plot(m_peak,m_freq,'rx',zorder=2)
            #plt.savefig('{}_time_compare.png'.format(STARTTIME)) 
            plt.show()
            #import pdb; pdb.set_trace()
            
            
            