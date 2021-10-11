#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:10:59 2020

This code detects fin or blue whale calls using spectrogram cross-correlation and
stores detection metrics in comma-separated variable files.

@author: wader
"""

import sys

sys.path.append('/Users/wader/Desktop/whaletracks/') 

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as siow
import numpy as np
import scipy.signal as sig
import matplotlib.colors as color
import detect_calls as detect
from whaletracks.detection.event_analyzer import EventAnalyzer
from whaletracks.common.util import datetimeToEpoch
import pandas as pd
import whaletracks.common.constants as cn
from datetime import datetime
from whaletracks.common.util import datetimeToEpoch
from scipy.signal import hilbert
import time
import math

FINFLAG = True #True if detecting fins, False if detecting blues
#CHUNK_FILE = "Blues_chunk_test.csv"
CHUNK_FILE = "test.csv" #"AlaskaFins_LA21_chunk.csv" #Name of saved call file
#FIN_DET_SERIES = "fin_series.csv"
PLOTFLAG = True #Use if troubleshooting and want to see plots.
MP_TS_FLAG = False #Use if storing Fin multipath info
MP_SPCT_FLAG = False
PLOTFLAG_STACK = False
CLIENT_CODE = 'IRIS'
network="YO" #Network name "OO" for OOI, "7D" for Cascadia, "XF" for marianas, "XO" for alaska, "YO" for ENAM
station= "X08" #"X06" for ENAM  "LD41",  "B19" for marianas station #Specific station, or '*' for all available stations
location='*'  # '*' for all available locations
channel= 'HHZ' #Choose channels,  you'll want 'BHZ,HHZ' for Cascadia
                      #Check http://ds.iris.edu/mda/OO/ for OOI station channels

#DET_PATH=cn.SCM_DETECTION.csv_path
DET_PATH="test.csv" #"AlaskaFins_LA21.csv" #Final save file

if FINFLAG == False:
    #Build blue whale B-call characteristics - wide
    F0 = 16 #average start frequency
    F1 = 14.3 #average end frequency
    BDWDTH = 1 # average bandwidth
    DUR = 15 #average duration

if FINFLAG:
    #Build fin whale call characteristics
    ######Low pulse kernel parameters##############
    #F0 = 20 #average start frequency
    #F1 = 15 #average end frequency
    #BDWDTH = 3 # average bandwidth
    #DUR = 1 #average duration

    ######High pulse kernel parameters##############
    #F0 = 30 #37 #average start frequency
    #F1 = 20 #average end frequency
    #BDWDTH = 5 # average bandwidth
    #DUR = 1 #average duration

     ######High pulse kernel parameters WD59##############
    #F0 = 32 #37 #average start frequency
    #F1 = 22 #average end frequency
    #BDWDTH = 3 # average bandwidth
    #DUR = 1 #average duration

    ######Marianas pulse kernel parameters##############
    #F0 = 20 #average start frequency
    #F1 = 15 #average end frequency
    #BDWDTH = 3 # average bandwidth
    #DUR = .8 #average duration

    ######Enam pulse kernel parameters##############
    #F0 = 26 #average start frequency
    #F1 = 17 #average end frequency
    #BDWDTH = 3 # average bandwidth
    #DUR = .8 #average duration

    ######Enam pulse kernel parameters MP##############
    F0 = 22 #average start frequency
    F1 = 18 #average end frequency
    BDWDTH = 3 # average bandwidth
    DUR = .8 #average duration

    ######Hawaii pulse kernel parameters##############
    #F0 = 25 #average start frequency
    #F1 = 15 #average end frequency
    #BDWDTH = 3 # average bandwidth
    #DUR = 1 #average duration

    ######Puerto Rico pulse kernel parameters##############
    #F0 = 36 #average start frequency
    #F1 = 25 #average end frequency
    #BDWDTH = 5 # average bandwidth
    #DUR = 1 #average duration



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

#STARTTIME = ("2015-12-10T23:00:00.000") # for fin testing puerto rico Kate Stafford
#ENDTIME = ("2015-12-11T02:00:00.000")

#STARTTIME = ("2010-09-16T00:00:00.000") #for Hawaii fins testing
#ENDTIME = ("2011-07-29T00:00:00.000")

#STARTTIME = ("2015-06-10T00:00:00.000") #for Puerto Rico fins full year
#ENDTIME = ("2015-12-10T00:00:00.000")

STARTTIME = ("2015-01-26T22:00:00.000") #for ENAM fins MP
ENDTIME = ("2015-01-26T22:59:00.000")

#STARTTIME = ("2015-02-17T23:00:00.000") #for ENAM fins MP X08 1
#ENDTIME = ("2015-02-17T23:59:00.000")

#STARTTIME = ("2015-02-09T02:00:00.000") #for ENAM fins testing 2
#ENDTIME = ("2015-02-09T10:00:00.000")

#STARTTIME = ("2019-02-11T03:00:00.000") #for Alaska fins testing
#ENDTIME = ("2019-02-11T08:00:00.000")

#STARTTIME = ("2011-01-24T10:30:00.000") #for Hawaii fins testing #2
#ENDTIME = ("2011-01-24T19:30:00.000")

#STARTTIME = ("2014-04-08T00:00:00.000") #for ENAM fins full year
#ENDTIME = ("2015-04-02T00:00:00.000")

#STARTTIME = ("2019-02-12T08:00:00.000") #for Alaska fins #2
#ENDTIME = ("2019-02-12T12:00:00.000")

#STARTTIME = ("2019-02-11T03:00:00.000") #for Alaska fins #1
#ENDTIME = ("2019-02-11T08:00:00.000")

#STARTTIME = ("2019-01-20T08:00:00.000") #for Alaska fins WD59 test
#ENDTIME = ("2019-01-01T01:00:00.000")
#ENDTIME = ("2019-01-20T11:00:00.000")

#STARTTIME = ("2019-02-16T15:00:00.000") #for Alaska fins WD62 test
#ENDTIME = ("2019-02-16T17:00:00.000")

#STARTTIME = ("2011-11-01T00:00:00.000") #for marianas fins
#ENDTIME = ("2012-04-01T00:00:00.000")

#STARTTIME = ("2012-01-09T04:10:00.000") # for blue whale freq testing FN14A
#ENDTIME = ("2012-01-09T04:20:00.000")

#STARTTIME = ("2012-03-30T20:00:00.000") # for fin max call testing marianas #1
#ENDTIME = ("2012-03-30T23:59:00.000")

#STARTTIME = ("2012-03-12T18:00:00.000") # for fin max call testing marianas #2
#ENDTIME = ("2012-03-12T21:59:00.000")

#STARTTIME = ("2018-10-25T13:07:00.000") #for testing on FN14A fins
#ENDTIME = ("2018-10-25T13:37:00.000")

HALF_HOUR = 1800  # in seconds
CHUNK_LENGTH=HALF_HOUR #secnods

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
        
    client = Client(client_code,user="rsharwade@rocketmail.com",password="EndgMk3hKe9J")
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
            utcstart_chunk=utcstart_chunk+CHUNK_LENGTH
            utcend_chunk=utcend_chunk+CHUNK_LENGTH
            continue

        try:
            #Remove sensitivity and response, and filter data    
            st_raw.detrend(type="demean")
            st_raw.detrend(type="linear")
            st_raw.remove_response(output='VEL',pre_filt=[1,3,48,49])
            st_raw.remove_sensitivity()
        except:
            #if fails, waits to reset connection then tries again
            print('Connection reset error, retrying')
            time.sleep(60*5)
            continue
               
        

        num_sta=len(st_raw) #count stations in trace
        analyzers_chunk=[] # initiate chunk dataframe
        #Run detector on each station
        for idx in range(1, num_sta+1):
    
            j = idx - 1
            tr=st_raw[j]
            
    
            tr_filt=tr.copy()
            #skip if less than 2 min of data
            if len(tr_filt.data) < tr_filt.stats.sampling_rate*60*2: 
                continue
            if tr_filt.data[0]==tr_filt.data[1]: #skip if data is bad (indicated by constant data)
                continue

            #Build detection metrics for either fin or blue whale calls
            if FINFLAG:

                #Spectrogram metrics
                window_size=.8
                overlap=.95
                freqlim=[10, 49]
                #SNR metrics
                snr_limits=[20, 49]
                snr_calllength=1
                snr_freqwidth=5
                #Event metrics
                prominence= 1000 #.2 #.5 #min threshold   .1 for 0.3 second window
                event_dur= .3 #minimum width of detection
                distance= 12 #minimum distance between detections
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
                prominence=.5 #min threshold
                event_dur= 5 #minimum width of detection
                distance=18 #minimum distance between detections
                rel_height=.7
                
             
            #Make spectrogram
            [f,t,Sxx]=detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=window_size, overlap=overlap, plotflag=False,filt_freqlim=freqlim,ylim=freqlim)
            
            #Make detection kernel
            [tvec, fvec, BlueKernel, freq_inds]=detect.buildkernel(f0, f1, bdwdth, dur, f, t, tr_filt.stats.sampling_rate, plotflag=False, kernel_lims=detect.finKernelLims)
            
            import pdb; pdb.set_trace()
            #subset spectrogram to be in same frequency range as kernel
            Sxx_sub=Sxx[freq_inds,:][0]
            f_sub=f[freq_inds]
            
            #Run detection using built kernel and spectrogram
            [times, values]=detect.xcorr(t,f_sub,Sxx_sub,tvec,fvec,BlueKernel, plotflag=PLOTFLAG,ylim=freqlim)
            
            #Pick detections using EventAnalyzer class
            analyzer_j = EventAnalyzer(times, values, utcstart_chunk, dur=event_dur, prominence=prominence, distance=distance, rel_height=rel_height)
            #analyzer_j.plot()

        
            
            dt_up=.6 #7
            dt_down=8 
            if MP_SPCT_FLAG and len(analyzer_j.df) > 1: #if MP_FLAG is True
                
                #Stack detected calls by adding spectrograms together
                [tstack,fstack,Sxxstack] = detect.stack_spect(t,f_sub,Sxx_sub,utcstart_chunk,analyzer_j,dt_up,dt_down)


            
                #Run detection kernel on stacked spectrogram
                [stacktimes, stackvalues]=detect.xcorr_log(tstack,fstack,Sxxstack,tvec,fvec,BlueKernel, plotflag=False,ylim=freqlim)

                #Autocorrelation of spectrogram and interpolation of score
                #[autotimes,autovals]=detect.spect_autocorr(t,f,Sxx,30,plotflag=PLOTFLAG,ylim=[15, 25])
                #timesnew=np.linspace(min(autotimes),max(autotimes),len(autotimes)*10)
                #f=interp1d(autotimes,autovals,kind='cubic')
                #valuesnew=f(timesnew)

                #detection score autocorrelation
                det_timesnew=np.linspace(min(times),max(times),len(times)*10)
                from scipy.interpolate import interp1d
                from scipy import signal
                #import pdb; pdb.set_trace()
                f=interp1d(times,values,kind='cubic')
                det_valuesnew1=f(det_timesnew)
                det_valuesnew=sig.detrend(det_valuesnew1,type='constant')
                corr=signal.correlate(det_valuesnew,det_valuesnew)
                #autocorr_chunk=corr[len(det_timesnew)-1:]/max(corr)
                autocorr_chunk=corr[len(det_timesnew)-math.ceil(dt_up/(det_timesnew[1]-det_timesnew[0])):len(det_timesnew)+math.ceil(dt_down/(det_timesnew[1]-det_timesnew[0]))]/max(corr)
                dettimes_chunk=det_timesnew[0:len(autocorr_chunk)]


                #Show interpolated autocorrelation for spectrogram and detection score
                #spectrogram
                mp_df = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                timesnew=np.linspace(min(stacktimes),max(stacktimes),len(stacktimes)*10)
                f=interp1d(stacktimes,stackvalues,kind='cubic')
                valuesnew=f(timesnew)

                #calculate multipaths for stacked spectrogram
                for det in range(0,len(analyzer_j.df)):
                    mp_event=analyzer_j.mp_picker(timesnew, valuesnew, utcstart_chunk, dur=.1, prominence=0)
                    mp_df=mp_df.append(mp_event,ignore_index=True)
                #analyzer_j.df= pd.concat([analyzer_j.df, mp_df], axis=1)

                #calculate multipaths for autocorrelated detection score
                mp_df_auto = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                dettimes_chunk=dettimes_chunk-min(dettimes_chunk)
                for det in range(0,len(analyzer_j.df)):
                    mp_event=analyzer_j.mp_picker(dettimes_chunk,autocorr_chunk, utcstart_chunk, dur=.1, prominence=0)
                    mp_df_auto=mp_df_auto.append(mp_event,ignore_index=True)
                #analyzer_j.df= pd.concat([analyzer_j.df, mp_df_auto], axis=1)
        


            #get multipath data for fins from timeseries
            if MP_TS_FLAG and len(analyzer_j.df) > 1: #if MP_FLAG is True
                mp_df_time = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                samples =list(range(0,len(tr_filt.data)))
                sos = sig.butter(4, np.array(snr_limits), 'bp', fs=tr_filt.stats.sampling_rate, output = 'sos') 
                filtered_data = sig.sosfiltfilt(sos, tr_filt.data)
                amplitude_envelope = abs(hilbert(filtered_data))
                #sos = sig.butter(30, 25, 'lp', fs=tr_filt.stats.sampling_rate, output = 'sos')
                #filtered_env = sig.sosfiltfilt(sos, amplitude_envelope)
                filtered_env = amplitude_envelope
                seconds=np.array([s/tr_filt.stats.sampling_rate for s in samples]) 

                utctimes=[utcstart_chunk + t for t in times]

                [stack_t,stack_env]=detect.stack_times(seconds,filtered_env,utcstart_chunk,analyzer_j,dt_up,dt_down)
                #import pdb; pdb.set_trace()

                #mp_df_time = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                #dettimes_chunk=dettimes_chunk-min(dettimes_chunk)
                for det in range(0,len(analyzer_j.df)):
                    mp_event=analyzer_j.mp_picker(stack_t,stack_env, utcstart_chunk, dur=.1, prominence=0)
                    mp_df_time=mp_df_time.append(mp_event,ignore_index=True)
                analyzer_j.df= pd.concat([analyzer_j.df, mp_df_time], axis=1)
                
                #for det in range(0,len(analyzer_j.df)):
                    #master_det = analyzer_j.df['start_time'][det]
                    #master_det_sec = (master_det - utcstart_chunk)*int(tr_filt.stats.sampling_rate)

                    #start_sample=max([master_det_sec-dt_up*tr_filt.stats.sampling_rate,min(samples)])
                    #end_sample=min([master_det_sec+dt_down*tr_filt.stats.sampling_rate,max(samples)])

                    #seq_samples = [s for s in samples if start_sample < s < end_sample]
                    #seq_seconds = [s/tr_filt.stats.sampling_rate for s in seq_samples]
                    #seq_data = filtered_env[seq_samples]

                    
                    #mp_event=analyzer_j.mp_picker(seq_seconds, seq_data, utcstart_chunk, prominence=10**-17)
                    #mp_df=mp_df.append(mp_event,ignore_index=True)
                    
                    
                   
                #import pdb; pdb.set_trace() 
                #analyzer_j.df= pd.concat([analyzer_j.df, mp_df], axis=1)

                #if PLOTFLAG:
                    #seconds=[s/tr_filt.stats.sampling_rate for s in samples]
                    #utctimes=[utcstart_chunk+s for s in seconds]
                    #plt.plot(utctimes,filtered_data)
                    #plt.plot(utctimes,filtered_env)
                    #plt.plot(seconds,filtered_data)
                    #plt.plot(seconds,filtered_env)
                    #plt.scatter(mp_df['arrival_1'],mp_df['amp_1'])
                    #plt.scatter(mp_df['arrival_2'],mp_df['amp_2'])
                    #plt.scatter(mp_df['arrival_3'],mp_df['amp_3'])
                    #plt.scatter(mp_df['arrival_4'],mp_df['amp_4'])
                    #plt.scatter(mp_df['arrival_5'],mp_df['amp_5'])
                    #plt.xlabel('seconds')
                    #plt.ylabel('amplitude')

                    #import pdb; pdb.set_trace()
                    #plt.show()

            if PLOTFLAG_STACK and len(analyzer_j.df) > 1:  #plots stacked spectrogram
                
                Sxx_log1=10*np.log10(Sxxstack)
                Sxx_log=Sxx_log1-np.min(Sxx_log1)
                #plt.figure(30, figsize=(9, 3))
                fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4,sharex=True)
                #fig=plt.figure()
                fig.set_figheight(8)
                fig.set_figwidth(12)
                
                ax0.plot(timesnew,valuesnew)
                ax0.scatter(mp_df['arrival_1'],mp_df['amp_1'])
                ax0.scatter(mp_df['arrival_2'],mp_df['amp_2'])
                ax0.scatter(mp_df['arrival_3'],mp_df['amp_3'])
                ax0.scatter(mp_df['arrival_4'],mp_df['amp_4'])
                ax0.scatter(mp_df['arrival_5'],mp_df['amp_5'])
                #ax0.set_xlabel('seconds')
                ax0.set_ylabel('amplitude')
                ax0.set_title('Detection score of stacked spectrogram ' + str(utcstart_chunk) + ' ' + tr_filt.stats.station)

                ax1.plot(dettimes_chunk,autocorr_chunk) #plot det score autocorrelation
                ax1.scatter(mp_df_auto['arrival_1'],mp_df_auto['amp_1'])
                ax1.scatter(mp_df_auto['arrival_2'],mp_df_auto['amp_2'])
                ax1.scatter(mp_df_auto['arrival_3'],mp_df_auto['amp_3'])
                ax1.scatter(mp_df_auto['arrival_4'],mp_df_auto['amp_4'])
                ax1.scatter(mp_df_auto['arrival_5'],mp_df_auto['amp_5'])
                #ax1.set_xlabel('Seconds')
                ax1.set_ylabel('autocorrelation')
                ax1.set_title('Autocorrelation of detection score')
                #ax1.set_xlim([0, 30])
                ax1.set_ylim([min(autocorr_chunk[130:]), max(autocorr_chunk[130:])])

                ax2.plot(stack_t,stack_env)
                ax2.scatter(mp_df_time['arrival_1'],mp_df_time['amp_1'])
                ax2.scatter(mp_df_time['arrival_2'],mp_df_time['amp_2'])
                ax2.scatter(mp_df_time['arrival_3'],mp_df_time['amp_3'])
                ax2.scatter(mp_df_time['arrival_4'],mp_df_time['amp_4'])
                ax2.scatter(mp_df_time['arrival_5'],mp_df_time['amp_5'])
                #ax0.set_xlabel('seconds')
                ax2.set_ylabel('timeseries envelope')
                ax2.set_title('Stacked timeseries')

                cmap = plt.get_cmap('magma')
                vmin=np.median(Sxx_log)+2*np.std(Sxx_log)
                vmax=np.median(Sxx_log)
                #vmin, vmax = scale_func(Sxx_log)
                norm = color.Normalize(vmin=vmin, vmax=vmax)
                #plt.subplot(212)
                im = ax3.pcolormesh(tstack, fstack, Sxx_log, cmap=cmap,norm=norm) 
                fig.colorbar(im, ax=ax2,orientation='horizontal')
                ax3.set_ylabel('Frequency [Hz]')
                #ax1.set_xticks([])
                ax3.set_xlabel('Time [seconds]')
                ax3.set_title('Stacked spectrogram')
                fig.tight_layout()
                #fig.savefig('Figures/Stack_autocorr_examples_LD41/window_0_5s/stackspect' + str(utcstart_chunk) + '.png', dpi=100)
                #plt.show()
                plt.close()
            #Calculate SNR info
            [snr,ambient_snr] = detect.get_snr(analyzer_j, t, f_sub, Sxx_sub, utcstart_chunk,snr_limits=snr_limits,snr_calllength=snr_calllength,snr_freqwidth=snr_freqwidth,dur=dur)

            #Add freq info for blue whales
            if FINFLAG == False:
                #These freqency metrics only work for blue whale calls
                [peak_freqs,start_freqs,end_freqs,peak_stds,start_stds,end_stds] = detect.freq_analysis(analyzer_j,t,f_sub,Sxx_sub,utcstart_chunk)
            if FINFLAG:
                #Fill frequency metric columns in csv with Nones if Fin calls
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
            analyzers_chunk.append(analyzer_j.df)

                      
        utcstart_chunk=utcstart_chunk+CHUNK_LENGTH
        utcend_chunk=utcend_chunk+CHUNK_LENGTH
        
        #Extend final dataframe with detections from current time chunk
        analyzers.extend(analyzers_chunk)
        #import pdb; pdb.set_trace()
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
