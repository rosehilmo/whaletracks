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
CHUNK_FILE='Fins_chunk_manual_HYSB1_0.5_complex.csv' #File name where verified calls will be saved
DETECTION_FILE='looking_for_calls_HYSB1_0.5_dist2.csv' #File name produced by main_detection

is_restart='True' #each iteration reads CHUNK_FILE and adds new verified calls
station_id_list=['HYSB1'] #List each station you want to verify

#Starttime_list = ["2020-11-15T22:00:00.000"] #high quality for testing
#create a list of time of detection

#For axial base
'''
Starttime_list = ["2015-12-16T13:00:00.000",
                  "2015-12-16T13:10:00.000",
                  "2015-12-16T13:20:00.000",
                  "2015-12-16T13:30:00.000",
                  "2015-12-16T13:40:00.000",
                  "2015-12-25T04:00:00.000",
                  "2015-12-25T04:10:00.000",
                  "2015-12-25T04:20:00.000",
                  "2015-12-25T04:30:00.000",
                  "2016-01-04T01:00:00.000",
                  "2016-01-04T01:10:00.000",
                  "2016-01-04T01:20:00.000",
                  "2016-01-04T01:30:00.000",
                  "2016-01-04T01:40:00.000",
                  "2020-09-25T02:00:00.000",
                  "2020-09-25T02:10:00.000",
                  "2020-09-25T02:20:00.000",
                  "2020-09-25T02:30:00.000",
                  "2020-09-25T02:40:00.000",
                  "2020-10-08T04:00:00.000",
                  "2020-10-08T04:10:00.000",
                  "2020-10-08T04:20:00.000",
                  "2020-10-08T04:30:00.000",
                  "2020-10-08T04:40:00.000",
                  "2020-11-26T07:00:00.000",
                  "2020-11-26T07:10:00.000",
                  "2020-11-26T07:20:00.000",
                  "2020-11-26T07:30:00.000",
                  "2020-11-26T07:40:00.000",
                  "2020-12-26T02:00:00.000",
                  "2020-12-26T02:10:00.000",
                  "2020-12-26T02:20:00.000",
                  "2020-12-26T02:30:00.000",
                  ]
'''

'''
#for hysb1
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

#for high quality at HYSB1
Starttime_list = ["2015-10-06T02:00:00.000",
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
                  "2016-11-20T15:10:00.000",
                  "2016-11-20T15:20:00.000",
                  "2017-01-01T06:20:00.000",
                  "2017-01-01T06:30:00.000",
                  "2017-01-01T06:40:00.000",
                  "2017-12-05T06:00:00.000",
                  "2017-12-05T06:10:00.000",
                  "2017-12-05T06:20:00.000",
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


for time_ind in range(0,len(Starttime_list)):
    STARTTIME = Starttime_list[time_ind] #start time

    STARTEPOCH = datestrToEpoch([STARTTIME],dateformat='%Y-%m-%dT%H:%M:%S.%f')
    ENDEPOCH = [STARTEPOCH[0] + CHUNK_LENGTH]

    utcstart_chunk = UTCDateTime(STARTTIME)
    utcend_chunk = UTCDateTime(STARTTIME) + CHUNK_LENGTH

    client=Client('IRIS')

    #Load call dataframes
    b_df=pd.read_csv(DETECTION_FILE)

    print(STARTTIME)
    #Either load working verification file or make a new one
    if os.path.isfile(CHUNK_FILE) and is_restart:
        verified_calls = pd.read_csv(CHUNK_FILE)
    else:
        verified_calls = pd.DataFrame(columns=b_df.columns)
        verified_calls['quality']=[]

    for station_ids in station_id_list:   
        st_raw_exist = False
        retry=0
        chunk_verified=pd.DataFrame(columns=b_df.columns)
        
        #Filter b, a and b filtered dataframes for specific station and date range
        b_df_sub=b_df.loc[(b_df['peak_epoch'] > STARTEPOCH[0]) &
                          (b_df['peak_epoch'] < ENDEPOCH[0])]

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
        [f,t,Sxx]=plotwav(tr_filt.stats.sampling_rate, tr_filt.data, filt_freqlim=[15, 25],window_size=5, overlap=.95, plotflag=False)
        t=t+STARTEPOCH 
        
        #Subsample spectrogram to call range and convert to dB
        freq_inds=np.where(np.logical_and(f>=12, f<=32))
        f_sub=f[freq_inds]
        Sxx_sub=Sxx[freq_inds,:][0]
        Sxx_log1=10*np.log10(Sxx_sub)
        Sxx_log=Sxx_log1-np.min(Sxx_log1)
        #Sxx_log = Sxx_sub
        #Choose color range for spectrogram
        vmin=np.median(Sxx_log)+1.5*np.std(Sxx_log)
        vmax=np.median(Sxx_log)

        if PLOTFLAG==True:
                
            t1=min(t)
            t2=max(t)
            #plot a and b calls on upper axis
            #plt.figure(PLT_SCORE, figsize=(9, 3))
            fig, (ax0) = plt.subplots(nrows=1,sharex=True)

        #plot spectrogram 
            cmap = plt.get_cmap('magma')
            norm = color.Normalize(vmin=vmin, vmax=vmax)
            #plt.subplot(212)
            im = ax0.pcolormesh(t, f_sub, Sxx_log, cmap=cmap,norm=norm) 
            fig.colorbar(im, ax=ax0,orientation='horizontal')
            ax0.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
            ax0.set_ylim([12, 32])
            ax0.set_ylabel('Frequency [Hz]')
            #ax1.set_xticks([])
            #ax1.set_xlabel('Time [seconds]')
            fig.tight_layout()
            
            #plot peak frequencies & time
            #plt.plot(b_peak,b_freq,'yx', zorder=1)
            
            
            
            #Request user input to select calls
            print('Select all calls')
            true_calls=plt.ginput(n=-1,timeout=-1)
        plt.savefig('{}.png'.format(STARTTIME))    
        plt.close()

        quality_list=[]
        with open(CHUNK_FILE,'a') as out:
            csv_out=csv.writer(out)
            #csv_out.writerow(['time','frequency'])
            for row in true_calls:
                csv_out.writerow(row)
        #import pdb; pdb.set_trace()
'''
        for inds2 in range(0,len(true_calls)):
            ind2 = inds2-1
            new_row=[None,None,None,None,None,true_calls[ind2][0],None,None,
                     None,None,None,None,None,None,None,None,None,None,station_ids,None,1]
            df_len=len(chunk_verified)
            chunk_verified.loc[df_len] = new_row

   

        verified_calls=verified_calls.append(chunk_verified, ignore_index=True)
        plt.close()
    
    verified_calls.to_csv(CHUNK_FILE,index=False)
'''