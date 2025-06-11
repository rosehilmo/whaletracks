#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:10:59 2024

This code detects Brydes whale calls and detects multipaths via autocorrelation of detection score

@author: wader
"""
import sys

sys.path.append('/Users/rhilmo/Documents/GitHub/whaletracks')

import pickle
import math
import time
from scipy.signal import hilbert
from datetime import datetime
import whaletracks.common.constants as cn
import pandas as pd
from whaletracks.common.util import datetimeToEpoch
from whaletracks.detection.event_analyzer import EventAnalyzer
import detect_calls as detect
import matplotlib.colors as color
import scipy.signal as sig
import numpy as np
import scipy.io.wavfile as siow
import os
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import basic_ranging_model as ranging
import datetime


PLOTFLAG = False  # Use if troubleshooting and want to see plots.
PLOTFLAG_STACK = False
CLIENT_CODE = 'IRIS'
CHUNK_FILE = "test.csv"
DET_PATH="test.csv"
network="XF" #Network name "OO" for OOI, "7D" for Cascadia, "XF" for PuertoRico, "XO" for alaska, "YO" for ENAM, "9A" for Hawaii, "ZZ" for puerto rico
station= "B19" #"LD41" # "B19" for PuertoRico station #Specific station, or '*' for all available stations "X06" for ENAM, "LOSW" for Hawaii, "XABV" for puerto rico
location='*'  # '*' for all available locations
channel= 'HHZ' #Choose channels,  you'll want 'BHZ,HHZ' for Cascadia 'ELZ' for hawaii
  


# Build fin whale call characteristics

######Mar pulse kernel parameters##############
#F0 = 22  # average start frequency
#F1 = 15  # average end frequency
#BDWDTH = 3  # average bandwidth
#DUR = .8  # average duration

######Brydes pulse kernel parameters WD50##############
F0 = 37 #average start frequency
F1 = 33 #average end frequency
BDWDTH = 2 # average bandwidth
DUR = 3 #average duration


STARTTIME = ("2012-02-03T00:00:00.000")  # oops
ENDTIME = ("2013-02-05T00:00:00.000")

# Download timeseries length
DAY_LENGTH = 60*60*12  # How long of a chunk to download at a time in seconds

# Chunk length for autocorrelation
CHUNK_LENGTH = 60*20  #How long of a chunk to use to estimate a range via autocorrelation
dt_up = 2
dt_down =9
# starttime=("2011-10-01T12:00:00.000")
# endtime=("2012-07-01T12:00:00.000")   



def main(STARTTIME, ENDTIME,
        client_code=CLIENT_CODE, f0=F0,
        f1=F1, bdwdth=BDWDTH, dur=DUR,
        detection_pth=DET_PATH,
        chunk_pth=CHUNK_FILE, station_ids=station,
        is_restart=True, dt_up = dt_up, dt_down = dt_down):
    """
    :param UTCDateTime starttime: ex. STARTTIME = ("2012-03-30T21:37:00.000")
    :param UTCDateTime endtime: ex. ENDTIME = ("2012-03-30T22:38:00.000")
    """

    if os.path.isfile(CHUNK_FILE) and is_restart: #checks to see if chunk files exist yet, loads if they do
        analyzers = [pd.read_csv(CHUNK_FILE)]
        auto_df_full = [pd.read_csv('auto_'+CHUNK_FILE)]
        #stack_df_full = [pd.read_csv('stack_Marianas_B19_v2.csv')]
    else:
        analyzers = [] #makes new variables if files do not exist
        auto_df_full = []
        #stack_df_full = []

    ar = np.empty((0, 205))
    # ar=np.empty((0,2251))
    datetimearray = []

    client = Client(client_code) #finds IRIS client code
    utcstart = UTCDateTime(STARTTIME) #finds datetime of start time
    utcend = UTCDateTime(ENDTIME) #finds datetime of end time

    utcstart_chunk = utcstart #sets start of chunk to download
    utcend_chunk = utcstart + DAY_LENGTH #sets end of chunk to download
    

    # Loop through times between starttime and endtime
    while utcend > utcstart_chunk: #loop through chunk times of data so long as the start chunk is less than the end time
        #import pdb; pdb.set_trace()
        print(utcstart_chunk)

        retry = 0
        st_raw_exist = False

        # Attempt to get waveforms
        while st_raw_exist == False and retry < 5: #tries up to 5 times to get timeseries data from IRIS
            try:
                st_raw = client.get_waveforms(network=network, station=station_ids, location=location,
                                            channel=channel, starttime=utcstart_chunk - .5*CHUNK_LENGTH,
                                            endtime=utcend_chunk + .5*CHUNK_LENGTH, attach_response=True)
                st_raw_exist = True
                retry = 5
            except: 
                retry = retry+1
                st_raw_exist = False
                print("Client failed: Retry " + str(retry) + " of 5 attempts")

        # Check if waveform data exists
        if st_raw_exist == False: 
            print("WARNING: no data available from input station/times") #prints warning message and moves on to next chunk if data retrieval is unsucessful
            utcstart_chunk = utcstart_chunk+DAY_LENGTH
            utcend_chunk = utcend_chunk+DAY_LENGTH
            continue

        try: 
            # Remove sensitivity and response, and filter data
            st_raw.detrend(type="demean")
            # st_raw.detrend(type="linear")
            st_raw.remove_response(output='VEL', pre_filt=[1, 3, 45, 49])
            # st_raw.remove_sensitivity()
        except:
            # if fails, waits to reset connection then tries again
            print('Connection reset error, retrying')
            time.sleep(60*5)
            continue

        num_sta = len(st_raw)  # count stations in trace
        analyzers_chunk = []  # initiate chunk dataframe
        # Run detector on each station
        for idx in range(1, num_sta+1): #iterates for each station (should only be 1 for this code)

            j = idx - 1
            tr = st_raw[j]

            tr_filt = tr.copy()
            # skip if less than 1 min of data
            if len(tr_filt.data) < tr_filt.stats.sampling_rate*59:
                continue
            # skip if data is bad (indicated by constant data)
            if tr_filt.data[0] == tr_filt.data[1]:
                continue


            # Build detection metrics for  calls

            # Spectrogram metrics
            window_size = .8
            overlap = .95
            freqlim = [10, 45]
            # SNR metrics
            snr_limits = [28, 40]
            snr_calllength = 1
            snr_freqwidth = 3
            # Event metrics
            prominence = 1000  # 4 #.2 #.5 #min threshold   .1 for 0.3 second window
            event_dur = .1  # minimum width of detection
            distance = 16  # minimum distance between detections
            rel_height = .5



            # Make spectrogram of entire timeseries
            [f, t, Sxx] = detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=window_size,
                                        overlap=overlap, plotflag=False, filt_freqlim=freqlim, ylim=freqlim)

            # Make detection kernel using spectrogram parameters
            [tvec, fvec, BlueKernel, freq_inds] = detect.buildkernel(
                f0, f1, bdwdth, dur, f, t, tr_filt.stats.sampling_rate, plotflag=PLOTFLAG, kernel_lims=detect.finKernelLims)

            # subset spectrogram to be in same frequency range as kernel
            Sxx_sub = Sxx[freq_inds, :][0]
            f_sub = f[freq_inds]

            # Run detection using built kernel and spectrogram
            #[times, values]=detect.xcorr_log(t,f_sub,Sxx_sub,tvec,fvec,BlueKernel, plotflag=PLOTFLAG,ylim=freqlim)

            # Run detection using built kernel and spectrogram
            [times, values] = detect.xcorr(
                t, f_sub, Sxx_sub, tvec, fvec, BlueKernel, plotflag=PLOTFLAG, ylim=freqlim)

            # Pick detections using EventAnalyzer class
            analyzer_j = EventAnalyzer(times, values, utcstart_chunk - .5*CHUNK_LENGTH, dur=event_dur,
                                    prominence=prominence, distance=distance, rel_height=rel_height)
            # analyzer_j.plot()

            #import pdb; pdb.set_trace()

            #[snr,ambient_snr,db_amps] = detect.get_snr(analyzer_j, t, f_sub, Sxx_sub, utcstart_chunk,snr_limits=snr_limits,snr_calllength=snr_calllength,snr_freqwidth=snr_freqwidth,dur=dur,fs=tr_filt.stats.sampling_rate,window_len=window_size)
            #import pdb; pdb.set_trace()

            if len(analyzer_j.df) >= 1:  # if there is at least one detected call in the entire day initiate multipath procedures
            #Measure amplitude and snr Brydes whale calls from time series
                samples = list(range(0, len(tr_filt.data)))
                # Design bandpass filter to look between SNR limits of call 
                sos = sig.butter(4, np.array(snr_limits), 'bp',
                                fs=tr_filt.stats.sampling_rate, output='sos')
                # filter timeseries with bandpass filter
                filtered_data = sig.sosfiltfilt(sos, tr_filt.data)
                # take hilbert envelope of timeseries
                amplitude_envelope = abs(hilbert(filtered_data))
                # calculate seconds
                seconds = np.array(
                    [s/tr_filt.stats.sampling_rate for s in samples])
                #measure amplitude and snr of calls in filtered time series 
                [maxamp, ambient_snr, snr, medamp] = detect.amps_snr_timeseries(
                    seconds, amplitude_envelope, utcstart_chunk - .5*CHUNK_LENGTH, analyzer_j, 4, 4, pad_length = 6000)  # get amplitudes in array
                
            #Measure amplitude and snr in band lower than Brydes whale calls to check for EQ and t-phases
                #Measure amplitude and snr from timeseries
                # Design bandpass filter to look between SNR limits of call (make these wide enough for both call types)
                sos_eq = sig.butter(4, np.array([5,15]), 'bp',
                                fs=tr_filt.stats.sampling_rate, output='sos')
                # filter timeseries with bandpass filter
                filtered_data_eq = sig.sosfiltfilt(sos_eq, tr_filt.data)
                # take hilbert envelope of timeseries
                amplitude_envelope_eq = abs(hilbert(filtered_data_eq))
                # calculate seconds
                #measure amplitude and snr of low frequency noise in filtered time series 
                [maxamp_eq, ambient_snr_eq, snr_eq, medamp_eq] = detect.amps_snr_timeseries(
                    seconds, amplitude_envelope_eq, utcstart_chunk - .5*CHUNK_LENGTH, analyzer_j, 4, 4, pad_length = 6000)  # get amplitudes in array



            else: #if no calls detected, make empty list
                snr=[];
                #import pdb; pdb.set_trace()
                continue

            #Fill frequency columns with Nones, we don't use these metrics for multipath ranging

            # Fill frequency metric columns in csv with Nones if Fin calls
            peak_freqs = list(np.repeat(None, len(snr)))
            start_freqs = list(np.repeat(None, len(snr)))
            end_freqs = list(np.repeat(None, len(snr)))
            peak_stds = list(np.repeat(None, len(snr)))
            start_stds = list(np.repeat(None, len(snr)))
            end_stds = list(np.repeat(None, len(snr)))
            #maxamps=list(np.repeat(None, len(snr)))

            # Make dataframe with detections and their features from current time chunk
            station_codes = np.repeat(
                tr_filt.stats.station, analyzer_j.df.shape[0])
            network_codes = np.repeat(
                tr_filt.stats.network, analyzer_j.df.shape[0])
            peak_epoch = datetimeToEpoch(analyzer_j.df['peak_time'])
            end_epoch = datetimeToEpoch(analyzer_j.df['end_time'])
            start_epoch = datetimeToEpoch(analyzer_j.df['start_time'])
            analyzer_j.df[cn.SNR] = snr
            analyzer_j.df['ambient_snr'] = ambient_snr
            analyzer_j.df['db_amps'] = 20*np.log10(maxamp)
            analyzer_j.df[cn.STATION_CODE] = station_codes
            analyzer_j.df[cn.NETWORK_CODE] = network_codes
            analyzer_j.df['peak_frequency'] = snr_eq
            analyzer_j.df['start_frequency'] = start_freqs
            analyzer_j.df['end_frequency'] = end_freqs
            analyzer_j.df['peak_frequency_std'] = peak_stds
            analyzer_j.df['start_frequency_std'] = start_stds
            analyzer_j.df['end_frequency_std'] = end_stds
            analyzer_j.df['peak_epoch'] = peak_epoch
            analyzer_j.df['start_epoch'] = start_epoch
            analyzer_j.df['end_epoch'] = end_epoch
            analyzers_chunk.append(analyzer_j.df)
            #analyzers_chunk = pd.concat([analyzers_chunk,analyzer_j.df])

            # begin multipath method
            data_starttime = tr_filt.stats.starttime
            utc_times = [data_starttime + ti for ti in times]
            timescount = int(round(CHUNK_LENGTH/(utc_times[1]-utc_times[0])))
            one_minute = int(60/(utc_times[1]-utc_times[0]))

            # find number of minutes for range calculation
            numchunks = round(((tr_filt.stats.endtime-tr_filt.stats.starttime)-CHUNK_LENGTH)/60)
            
            #import pdb; pdb.set_trace()

            for mp_iter in range(numchunks): #iterate for each minute in day, calculates a multipath range for each minute that there are calls
                print(mp_iter)
                startind = one_minute*mp_iter #finds start and end indexes in series of chunk length used for autocorrelation 
                endind = one_minute*mp_iter+timescount

                times_sub = times[startind:endind] #indexes times and detection score to extract each chunk 
                values_sub = values[startind:endind]

                #import pdb; pdb.set_trace()
                start_utc=utc_times[startind]
                end_utc=utc_times[endind]

                j_df_sub=analyzer_j.df.loc[(analyzer_j.df['peak_time'] >= start_utc) & 
                        (analyzer_j.df['peak_time'] < end_utc)] #Find all detection in chunk

                min_df_sub=analyzer_j.df.loc[(analyzer_j.df['peak_time'] >= (start_utc+CHUNK_LENGTH/2-30)) & 
                        (analyzer_j.df['peak_time'] < (end_utc-CHUNK_LENGTH/2+30))] #find all detections in center minute of chunk


                #datetimearray = datetimearray + [utcstart_chunk] 
                #import pdb; pdb.set_trace()
                if len(min_df_sub) >= 1 and len(j_df_sub) >= 3:  #If there is at least one call in the center minute, calculate multipath range for the minute
                    """
                    #Stack detected calls by adding spectrograms together
                    [tstack,fstack,Sxxstack] = detect.stack_spect(t,f_sub,Sxx_sub,utcstart_chunk,analyzer_j,dt_up,dt_down)


                    #Run detection kernel on stacked spectrogram
                    [stacktimes, stackvalues]=detect.xcorr(tstack,fstack,Sxxstack,tvec,fvec,BlueKernel, plotflag=False,ylim=freqlim)
                    """
                    #import pdb; pdb.set_trace()
                    # detection score autocorrelation
                    det_timesnew = np.linspace(
                        min(times_sub), max(times_sub), len(times_sub)*10) #increase resolution of time values by 10x
                    from scipy.interpolate import interp1d
                    from scipy import signal
                    #Increase resolution of detection score series by cubic spline interpolation
                    f = interp1d(times_sub, values_sub, kind='cubic')
                    det_valuesnew1 = f(det_timesnew)
                    det_valuesnew = sig.detrend(det_valuesnew1, type='constant')
                    #Autocorrelate hig-resolution detection score values
                    corr = signal.correlate(det_valuesnew, det_valuesnew)
                    #Take center chunk of autocorrelated series that corresponds to search window
                    autocorr_chunk = corr[len(det_timesnew)-math.ceil(dt_up/(det_timesnew[1]-det_timesnew[0])):len(
                        det_timesnew)+math.ceil(dt_down/(det_timesnew[1]-det_timesnew[0]))]/max(corr)
                    dettimes_chunk = det_timesnew[0:len(autocorr_chunk)]

                    """
                    #Show interpolated autocorrelation for spectrogram and detection score
                    #spectrogram
                    mp_df = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                    timesnew=np.linspace(min(stacktimes),max(stacktimes),len(stacktimes)*10)
                    f=interp1d(stacktimes,stackvalues,kind='cubic')
                    valuesnew=f(timesnew)

                    #calculate multipaths for stacked spectrogram
                    for det in range(0,len(analyzer_j.df)):
                        mp_event=analyzer_j.mp_picker(timesnew, valuesnew, utcstart_chunk, dur=.01, prominence=0,distance=2,rel_height=.5)
                        mp_df=mp_df.append(mp_event,ignore_index=True)
                    #analyzer_j.df= pd.concat([analyzer_j.df, mp_df], axis=1)
                    #import pdb; pdb.set_trace()
                    d1 = {'date': [utcstart_chunk+CHUNK_LENGTH/2], 'epoch': datetimeToEpoch([utcstart_chunk+CHUNK_LENGTH/2]), 'n_calls': [len(mp_df)], 'peaks': [np.median(analyzer_j.df['peak_signal'])], 'snr': [np.median(analyzer_j.df['snr'])], 'db_amps': [np.median(analyzer_j.df['db_amps'])]}
                    stack_df=pd.DataFrame(d1)
                    stack_df=pd.concat([stack_df, mp_df.head(1)], axis=1)
                    stack_df_full.append(stack_df)
                    new_stack_df=pd.concat(stack_df_full)
                    """

                    # detect timings of multipaths using autocorrelated detection score
                    mp_df_auto = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns) # makes new autocorrelation dataframe
                    dettimes_chunk = dettimes_chunk-min(dettimes_chunk) 
                    mp_event = analyzer_j.mp_picker(
                            dettimes_chunk, autocorr_chunk, start_utc, dur=.1, prominence=0, distance=1, rel_height=.5) #picks peaks in autocorrelation score to find multipath timings
                    for det in range(0, len(min_df_sub)):
                        
                        #mp_df_auto = mp_df_auto.append(mp_event, ignore_index=True)
                        mp_df_auto = pd.concat([mp_event, mp_df_auto], ignore_index = True)
                    
                    #import pdb; pdb.set_trace()
                    min_df_sub=min_df_sub.reset_index(drop=True)
                    mp_df_auto=mp_df_auto.reset_index(drop=True)
                    min_df_sub = pd.concat([min_df_sub, mp_df_auto], axis=1)
                    #Record metrics included in each autocorrelation minute including the datetime, number of calls in minute, number of calls in autocorrelation chunk, average detection score, snr, and amplitudes
                    d2 = {'date': [start_utc+CHUNK_LENGTH/2], 'epoch': datetimeToEpoch([start_utc+CHUNK_LENGTH/2]), 'n_calls': [len(mp_df_auto)], 'sum_calls': [len(j_df_sub)], 'peaks': [
                        np.median(j_df_sub['peak_signal'])], 'snr': [np.median(j_df_sub['snr'])], 'db_amps': [np.median(j_df_sub['db_amps'])], 'low_snr': [np.mean(j_df_sub['peak_frequency'])]}
                    auto_df = pd.DataFrame(d2)
                    auto_df = pd.concat([auto_df, mp_df_auto.head(1)], axis=1)
                    #import pdb; pdb.set_trace()
                    # make stack and auto dataframes
                    auto_df_full.append(auto_df)
                    #auto_df_full= pd.concat([auto_df_full,auto_df])
                
                    
                    #new_stack_df.to_csv('stack_Marianas_B19_v2.csv', index=False)


                # plots stacked spectrogram: NOT CURRENTLY IN USE SINCE WE DO NOT USE STACKING METHOD
                if PLOTFLAG_STACK and len(analyzer_j.df) >= 1:

                    Sxx_log1 = 10*np.log10(Sxxstack)
                    Sxx_log = Sxx_log1-np.min(Sxx_log1)
                    #plt.figure(30, figsize=(9, 3))
                    fig, (ax0, ax1, ax3) = plt.subplots(nrows=3, sharex=True)
                    # fig=plt.figure()
                    fig.set_figheight(8)
                    fig.set_figwidth(12)

                    ax0.plot(timesnew, valuesnew)
                    ax0.scatter(mp_df['arrival_1'], mp_df['amp_1'])
                    ax0.scatter(mp_df['arrival_2'], mp_df['amp_2'])
                    ax0.scatter(mp_df['arrival_3'], mp_df['amp_3'])
                    ax0.scatter(mp_df['arrival_4'], mp_df['amp_4'])
                    ax0.scatter(mp_df['arrival_5'], mp_df['amp_5'])
                    # ax0.set_xlabel('seconds')
                    ax0.set_ylabel('amplitude')
                    ax0.set_title('Detection score of stacked spectrogram ' +
                                str(utcstart_chunk) + ' ' + tr_filt.stats.station)

                    # plot det score autocorrelation
                    ax1.plot(dettimes_chunk, autocorr_chunk)
                    ax1.scatter(mp_df_auto['arrival_1'], mp_df_auto['amp_1'])
                    ax1.scatter(mp_df_auto['arrival_2'], mp_df_auto['amp_2'])
                    ax1.scatter(mp_df_auto['arrival_3'], mp_df_auto['amp_3'])
                    ax1.scatter(mp_df_auto['arrival_4'], mp_df_auto['amp_4'])
                    ax1.scatter(mp_df_auto['arrival_5'], mp_df_auto['amp_5'])
                    # ax1.set_xlabel('Seconds')
                    ax1.set_ylabel('autocorrelation')
                    ax1.set_title('Autocorrelation of detection score')
                    #ax1.set_xlim([0, 30])
                    ax1.set_ylim([min(autocorr_chunk[130:]),
                                max(autocorr_chunk[130:])])

                    cmap = plt.get_cmap('magma')
                    vmin = np.median(Sxx_log)+2*np.std(Sxx_log)
                    vmax = np.median(Sxx_log)
                    #vmin, vmax = scale_func(Sxx_log)
                    norm = color.Normalize(vmin=vmin, vmax=vmax)
                    # plt.subplot(212)
                    im = ax3.pcolormesh(tstack, fstack, Sxx_log,
                                        cmap=cmap, norm=norm)
                    fig.colorbar(im, ax=ax1, orientation='horizontal')
                    ax3.set_ylabel('Frequency [Hz]')
                    # ax1.set_xticks([])
                    ax3.set_ylim((freqlim[0], freqlim[1]))
                    ax3.set_xlabel('Time [seconds]')
                    ax3.set_title('Stacked spectrogram')
                    fig.tight_layout()
                    fig.savefig('Figures/Marianas/window_0_8s/stackspect' +
                                str(utcstart_chunk) + '.png', dpi=100)
                    # plt.show()
                    plt.close()
                # Calculate SNR info

        
            # Extend final dataframe with detections from current time chunk
            analyzers.extend(analyzers_chunk) #extends detection dataframe with current chunk detections

            #import pdb; pdb.set_trace()
            try: #write detection and autocorrelation multipaths to csvs for each day loop
                new_df = pd.concat(analyzers)
                new_df.to_csv(chunk_pth, index=False)
                new_auto_df = pd.concat(auto_df_full)
                new_auto_df.to_csv('auto_'+CHUNK_FILE, index=False)
                #new_stack_df.to_csv('stack_'+CHUNK_FILE, index=False)
            except:
                continue

        utcstart_chunk = utcstart_chunk+DAY_LENGTH #move to next day 
        utcend_chunk = utcend_chunk+DAY_LENGTH


    if len(analyzers) == 0: #write final detections dataframe after entire year is done looping
        print('WARNING: detections dataframe empty')
        final_analyzer_df = []
    elif len(analyzer_j.df) == 0:
        print('WARNING: no detections from current time window')
        #final_analyzer_df = []
    else:
        
        final_analyzer_df = pd.concat(analyzers)
        final_analyzer_df.to_csv(detection_pth, index=False)
   

###########Code parameters start here#############

sta_table = pd.read_csv('Station_info_Marianas_Brydes.csv',parse_dates=['startdate','enddate'])
#sta_table['startdate'] = sta_table['startdate']+datetime.timedelta(days=2)    




for site_ind in range(0,7): #range(13,len(sta_table['Sites'])):
    network = "XF"  # Network name "OO" for OOI, "7D" for Cascadia, "XF" for PuertoRico, "XO" for alaska, "YO" for ENAM, "9A" for Hawaii, "ZZ" for puerto rico
    station = sta_table['Sites'][site_ind]    # "LD41" # "B19" for PuertoRico station #Specific station, or '*' for all available stations "X06" for ENAM, "LOSW" for Hawaii, "XABV" for puerto rico
    location = '*'  # '*' for all available locations
    channel = sta_table['Channel'][site_ind]   # Choose channels,  you'll want 'BHZ,HHZ' for Cascadia 'ELZ' for hawaii
    depth = sta_table['Instrument Depth (m)'][site_ind]

    ss=1512
    sed_speed = 1512
    t = sta_table['Reflector depth (m)'][site_ind] - depth #estimates thickness of sediment layer

    [distance,t0,t1,t2,t3,t0_1,t1_1,t1_2]=ranging.basic_ranging(depth,ss,sed_speed,t,plotflag=False) #calculates estimated timings of arrivals using straight geometrical raypaths
    max_diff=max(np.subtract(t1_1,t0))
    dt_down = max_diff + 0.2 #sets MP search window based on maximum difference between multipaths + a small buffer
    dt_up=2

    CHUNK_FILE = station+'_mp_Brydes_20min.csv'  #file name changes depending on station
    DET_PATH = CHUNK_FILE
    STARTTIME = sta_table['startdate'][site_ind] #sets start date based on Station Info csv
    ENDTIME = sta_table['enddate'][site_ind] #sets end date based on Station Info csv

    

    main(STARTTIME, ENDTIME,
        client_code=CLIENT_CODE, f0=F0,
        f1=F1, bdwdth=BDWDTH, dur=DUR,
        detection_pth=DET_PATH,
        chunk_pth=CHUNK_FILE, station_ids=station,
        is_restart=True, dt_up = dt_up, dt_down = dt_down)





#if __name__ == "__main__":
#    main(STARTTIME, ENDTIME)
