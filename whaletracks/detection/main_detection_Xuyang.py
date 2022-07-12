#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:10:59 2020

This code detects fin or blue whale calls using spectrogram cross-correlation and
stores detection metrics in comma-separated variable files.

@author: wader
"""

import sys

sys.path.append('/Users/wader/Documents/GitHub/whaletracks/') 

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as siow
import numpy as np
import scipy.signal as sig
import matplotlib.colors as color
import detect_calls_Xuyang as detect
from event_analyzer_xuyang import EventAnalyzer
from whaletracks.common.util import datetimeToEpoch
import pandas as pd
import whaletracks.common.constants as cn
from datetime import datetime
from scipy.signal import hilbert
import time

FINFLAG = True #True if detecting fins, False if detecting blues
#CHUNK_FILE = "Blues_chunk_test.csv"
CHUNK_FILE = "Fins_chunk_test_peak.csv" #Name of saved call file
#FIN_DET_SERIES = "fin_series.csv"
PLOTFLAG = True  #Use if troubleshooting and want to see plots.
MP_FLAG = False #Use if storing Fin multipath info
CLIENT_CODE = 'IRIS'
network="OO" #Network name "OO" for OOI, "7D" for Cascadia, "XF" for marianas
station= "HYSB1" # "B19" for marianas station #Specific station, or '*' for all available stations
location='*'  # '*' for all available locations
channel= 'HHZ' #Choose channels,  you'll want 'BHZ,HHZ' for Cascadia
                      #Check http://ds.iris.edu/mda/OO/ for OOI station channels

#DET_PATH=cn.SCM_DETECTION.csv_path
DET_PATH="looking_for_calls_test_peak.csv" #Final save file
NEVENT_PATH='No_event_log_macair.csv' #file recording time where no data collected

if FINFLAG == False:
    #Build blue whale B-call characteristics - wide
    F0 = 16 #average start frequency
    F1 = 14.3 #average end frequency
    BDWDTH = 1 # average bandwidth
    DUR = 15 #average duration

if FINFLAG:
    #Build fin whale call characteristics
    F0 = 27 #average start frequency
    F1 = 19 #average end frequency
    BDWDTH = 5 # average bandwidth
    DUR = 0.7#average duration

#Blue whale A-call characteristics
#F0 = 14.5 #average start frequency
#F1 = 14.2 #average end frequency
#BDWDTH = .5 # average bandwidth
#DUR = 20 #average duration

#Blue whale B-call characteristics - narrow
#F0 = 15.7 #average start frequency
#F1 = 14.6 #average end frequency
#BDWDTH = .7 # average bandwidth
#DUR = 10 #average duration

#STARTTIME = ("2015-09-01T01:30:00.000") # for testing_xuyang
#ENDTIME =   ("2015-09-01T02:10:00.000")

STARTTIME = ("2019-10-15T07:00:00.000") # for testing_xuyang @axab1
ENDTIME =   ("2019-10-15T07:20:00.000")

#STARTTIME = ("2012-02-02T00:00:00.000") #for marianas fins
#ENDTIME = ("2013-02-06T00:00:00.000")

#STARTTIME = ("2012-01-09T04:10:00.000") # for blue whale freq testing FN14A
#ENDTIME = ("2012-01-09T04:20:00.000")

#STARTTIME = ("2012-03-30T21:37:00.000") # for fin max call testing marianas
#ENDTIME = ("2012-03-30T21:40:00.000")

#STARTTIME = ("2018-10-25T13:07:00.000") #for testing on FN14A fins
#ENDTIME = ("2018-10-25T13:37:00.000")

HALF_HOUR = 1800  # in seconds
CHUNK_LENGTH=HALF_HOUR/3 #secnods

#starttime=("2011-10-01T12:00:00.000")
#endtime=("2012-07-01T12:00:00.000")

#Fin instruments: network ="XF" and station="B19"

def main(STARTTIME, ENDTIME,
         client_code=CLIENT_CODE, f0=F0,
         f1=F1,bdwdth=BDWDTH,dur=DUR,
         detection_pth=DET_PATH, 
         chunk_pth=CHUNK_FILE,station_ids=station,
         is_restart=True):
    """
    :param UTCDateTime starttime: ex. STARTTIME = ("2012-03-30T21:37:00.000")
    :param UTCDateTime endtime: ex. ENDTIME = ("2012-03-30T22:38:00.000")
    """

    if os.path.isfile(CHUNK_FILE) and is_restart:
        analyzers = [pd.read_csv(CHUNK_FILE)]
    else:
        analyzers = []
        
    client = Client(client_code)
    utcstart = UTCDateTime(STARTTIME)
    utcend = UTCDateTime(ENDTIME)

    utcstart_chunk = utcstart
    utcend_chunk = utcstart + CHUNK_LENGTH
    
    #Loop through times between starttime and endtime
    while utcend > utcstart_chunk:
        #import pdb; pdb.set_trace()
        print(utcstart_chunk)
        
        retry=0
        st_raw_exist=False

        #Attempt to get waveforms
        while st_raw_exist == False and retry < 5:
            try:
                st_raw=client.get_waveforms(network=network, station=station_ids, location=location,
                                            channel=channel, starttime=utcstart_chunk,
                                            endtime=utcend_chunk, attach_response=True)
                st_raw_exist=True
                retry=5
            except:
                retry=retry+1
                st_raw_exist=False
                print("Client failed: Retry " + str(retry) + " of 5 attempts")
                
        #Check if waveform data exists        
        if st_raw_exist == False:
            print("WARNING: no data available from input station/times")
            #write to a csv file
            N_event=pd.read_csv(NEVENT_PATH)
            N_event.loc[len(N_event.index)] = [(utcstart_chunk),(station_ids)]
            N_event.to_csv(NEVENT_PATH, index=False)
            
            utcstart_chunk=utcstart_chunk+CHUNK_LENGTH
            utcend_chunk=utcend_chunk+CHUNK_LENGTH
            continue

        try:
            #Remove sensitivity and response, and filter data    
            st_raw.detrend(type="demean")
            st_raw.detrend(type="linear")
            st_raw.remove_response(output='VEL',pre_filt=[1,3,40,45])
            st_raw.remove_sensitivity()
        except:
            #if fails, waits to reset connection then tries again
            print('Connection reset error, retrying')
            time.sleep(60*5)
            continue
               
        

        num_sta=len(st_raw)
        analyzers_chunk=[]
        #Run detector on each station
        for idx in range(1, num_sta+1):
    
            j = idx - 1
            tr=st_raw[j]
            
    
            tr_filt=tr.copy()
            
            if len(tr_filt.data) < tr_filt.stats.sampling_rate*60*5: #skip if less than 1 min of data
                continue
            if tr_filt.data[0]==tr_filt.data[1]: #skip if data is bad (indicated by constant data)
                continue



            #Build detection metrics for either fin or blue whale calls
            if FINFLAG:

                #Spectrogram metrics
                window_size=1
            
                overlap=.95
                freqlim=[3, 37]
                #SNR metrics
                snr_limits=[15, 25]
                snr_calllength=1
                snr_freqwidth=5
                #Event metrics
                prominence=1000 #min threshold
                event_dur= .1 #minimum width of detection
                distance=15 #minimum distance between detections
                rel_height=.8



            if FINFLAG == False:
                #Spectrogram metrics
                window_size=5
                overlap=.95
                freqlim=[10, 20]
                #SNR metrics
                snr_limits=[14, 16]
                snr_calllength=4
                snr_freqwidth=.6
                #Event metrics
                prominence= 1#min threshold
                event_dur= 5 #minimum width of detection
                distance=5 #minimum distance between detections
                rel_height=.7
                
             
            #Make spectrogram
            [f,t,Sxx]=detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=window_size, overlap=overlap, plotflag=PLOTFLAG,filt_freqlim=freqlim,ylim=freqlim)
            #import pdb; pdb.set_trace()
            #Make detection kernel
            [tvec, fvec, BlueKernel, freq_inds]=detect.buildkernel(f0, f1, bdwdth, dur, f, t, tr_filt.stats.sampling_rate, plotflag=PLOTFLAG, kernel_lims=detect.defaultKernelLims)
            
            #subset spectrogram to be in same frequency range as kernel
            Sxx_sub=Sxx[freq_inds,:][0]
            f_sub=f[freq_inds]
            
            #Run detection using built kernel and spectrogram
            [times, values]=detect.xcorr(t,f_sub,Sxx_sub,tvec,fvec,BlueKernel, plotflag=PLOTFLAG,ylim=freqlim)
            
           #Pick detections using EventAnalyzer class
            analyzer_j = EventAnalyzer(times, values, utcstart_chunk, dur=event_dur, prominence=prominence, distance=distance, rel_height=rel_height)
            #analyzer_j.plot()
            
            
            #find slope of calls using peak in detection score
            #[times, values]=detect.finding_slope(t,f_sub,Sxx_sub,tvec,fvec,BlueKernel, plotflag=PLOTFLAG,ylim=freqlim)
            
                

            #Calculate SNR info
            [snr,ambient_snr] = detect.get_snr(analyzer_j, t, f_sub, Sxx_sub, utcstart_chunk,snr_limits=snr_limits,snr_calllength=snr_calllength,snr_freqwidth=snr_freqwidth,dur=dur)

            #Add freq info for fin whales
            if FINFLAG == True:
                #These freqency metrics only work for fin whale calls
                [peak_freqs,start_freqs,end_freqs,peak_stds,start_stds,end_stds] = detect.freq_analysis(analyzer_j,t,f_sub,Sxx_sub,utcstart_chunk)
                #[peak_freqs,start_freqs,end_freqs,peak_stds,start_stds,end_stds] = detect.freq_by_sum(analyzer_j,t,f_sub,Sxx_sub,utcstart_chunk)
                
            if FINFLAG == False:
                #Fill frequency metric columns in csv with Nones if Blue calls
                peak_freqs=list(np.repeat(None, len(snr)))
                start_freqs=list(np.repeat(None, len(snr)))
                end_freqs=list(np.repeat(None, len(snr)))
                peak_stds=list(np.repeat(None, len(snr)))
                start_stds=list(np.repeat(None, len(snr)))
                end_stds=list(np.repeat(None, len(snr)))

            #Make dataframe with detections from current time chunk
            station_codes = np.repeat(tr_filt.stats.station,analyzer_j.df.shape[0])
            network_codes = np.repeat(tr_filt.stats.network,analyzer_j.df.shape[0])
            peak_epoch=datetimeToEpoch(analyzer_j.df['peak_time'])
            end_epoch=datetimeToEpoch(analyzer_j.df['end_time'])
            start_epoch=datetimeToEpoch(analyzer_j.df['start_time'])
            analyzer_j.df[cn.STATION_CODE] = station_codes
            analyzer_j.df[cn.NETWORK_CODE] = network_codes
            analyzer_j.df[cn.SNR] = snr
            analyzer_j.df['ambient_snr'] = ambient_snr
            analyzer_j.df['peak_frequency'] = peak_freqs
            analyzer_j.df['start_frequency'] = start_freqs
            analyzer_j.df['end_frequency'] = end_freqs
            analyzer_j.df['peak_frequency_std'] = peak_stds
            analyzer_j.df['start_frequency_std'] = start_stds
            analyzer_j.df['end_frequency_std'] = end_stds
            analyzer_j.df['peak_epoch']=peak_epoch
            analyzer_j.df['start_epoch']=start_epoch
            analyzer_j.df['end_epoch']=end_epoch
            #analyzers_chunk.append(analyzer_j.df)

            if len(analyzer_j.df) >= 1: 
                #mp_df_time = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                samples =list(range(0,len(tr_filt.data)))
                sos = sig.butter(4, np.array(snr_limits), 'bp', fs=tr_filt.stats.sampling_rate, output = 'sos') #Design bandpass filter to look between SNR limits of call (make these wide enough for both call types)
                filtered_data = sig.sosfiltfilt(sos, tr_filt.data) #filter timeseries with bandpass filter
                amplitude_envelope = abs(hilbert(filtered_data)) #take hilbert envelope of timeseries
                seconds=np.array([s/tr_filt.stats.sampling_rate for s in samples]) #calculate seconds

                maxamps=detect.get_amps_max(seconds,amplitude_envelope,utcstart_chunk,analyzer_j,1,2) #get amplitudes in array
                analyzer_j.df['peak_amp']=maxamps

                medamps=np.median(np.abs(filtered_data))
                analyzer_j.df['med_amp']=[medamps]*len(maxamps)
            
                medamps_prec=detect.get_amps_med(seconds,filtered_data,utcstart_chunk,analyzer_j,5,-2) #get amplitudes in array
                analyzer_j.df['preceding_amp']=medamps

                
            

            analyzers_chunk.append(analyzer_j.df)   
                      
        utcstart_chunk=utcstart_chunk+CHUNK_LENGTH
        utcend_chunk=utcend_chunk+CHUNK_LENGTH
        
        #Extend final dataframe with detections from current time chunk
        analyzers.extend(analyzers_chunk)
        new_df = pd.concat(analyzers)
        new_df.to_csv(chunk_pth, index=False)
        

    
    if len(analyzers) == 0:
        print('WARNING: detections dataframe empty')
        final_analyzer_df = []
        
    else:
        final_analyzer_df = pd.concat(analyzers)
        final_analyzer_df.to_csv(detection_pth,index=False)
    #import pdb; pdb.set_trace()    
    return final_analyzer_df
    
 

if __name__ == "__main__":
    main(STARTTIME, ENDTIME)
