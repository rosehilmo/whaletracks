#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% CORTADO Affiliated Script %%%%%%%%%
%%%%%%%%%%% Required Information %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Created on Thu Jan  9 11:10:59 2020

This code detects fin whale calls using spectrogram cross-correlation and 
measures the timings of multipath arrivals that reflect from the sea floor and surface.
Code stores individual call detections and multipath arrival timings in comma-separated variable files.

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
MP_TS_FLAG = True  # Use if storing Fin multipath info
MP_SPCT_FLAG = True
PLOTFLAG_STACK = False
CLIENT_CODE = 'IRIS'
CHUNK_FILE = "test.csv"
DET_PATH="test.csv"
network="XF" #Network name "OO" for OOI, "7D" for Cascadia, "XF" for PuertoRico, "XO" for alaska, "YO" for ENAM, "9A" for Hawaii, "ZZ" for puerto rico
station= "B19" #"LD41" # "B19" for PuertoRico station #Specific station, or '*' for all available stations "X06" for ENAM, "LOSW" for Hawaii, "XABV" for puerto rico
location='*'  # '*' for all available locations
channel= 'HHZ' #Choose channels,  you'll want 'BHZ,HHZ' for Cascadia 'ELZ' for hawaii
  

# Build fin whale call characteristics

######Example kernel parameters for calls recorded in 2012-2013 in Marianas region##############
F0 = 22  # average start frequency
F1 = 15  # average end frequency
BDWDTH = 3  # average bandwidth
DUR = .8  # average duration


STARTTIME = ("2012-02-03T00:00:00.000")  # oops
ENDTIME = ("2013-02-05T00:00:00.000")


HALF_HOUR = 1800  # in seconds
# CHUNK_LENGTH=HALF_HOUR/5 #secnods

# Download timeseries length
DAY_LENGTH = 60*60*12  # Seconds

# Chunk length for autocorrelation
CHUNK_LENGTH = 60*10  # seconds

INSTRUMENT_DEPTH = 6000 #estimated instrument depth used to limit max multipath search window

dt_up = .6  #Search window starts 0.6 seconds before main detection
dt_down = (INSTRUMENT_DEPTH*2)/1500 + 0.5 #Search window ends 0.5 seconds later than the maximum two-way travel time at c=1500 m/s
    

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

    if os.path.isfile(CHUNK_FILE) and is_restart: #opens previous detection and multipath timing file if they exist
        analyzers = [pd.read_csv(CHUNK_FILE)]
        try:
            auto_df_full = [pd.read_csv('auto_'+CHUNK_FILE)]
        except:
            auto_df_full = []

    else: #starts a new detection and multipath timing file
        analyzers = []
        auto_df_full = []
        #stack_df_full = []


    #Set client and times
    client = Client(client_code) 
    utcstart = UTCDateTime(STARTTIME)
    utcend = UTCDateTime(ENDTIME)

    utcstart_chunk = utcstart
    utcend_chunk = utcstart + DAY_LENGTH
    

    # Loop through times between starttime and endtime
    while utcend > utcstart_chunk:
        #import pdb; pdb.set_trace()
        print(utcstart_chunk)

        retry = 0
        st_raw_exist = False

        # Attempt to get waveforms
        while st_raw_exist == False and retry < 5:
            try: #request data from IRIS using OBSpy toolbox
                st_raw = client.get_waveforms(network=network, station=station_ids, location=location,
                                            channel=channel, starttime=utcstart_chunk - .5*CHUNK_LENGTH,
                                            endtime=utcend_chunk + .5*CHUNK_LENGTH, attach_response=True)
                st_raw_exist = True
                retry = 5
            except: #sometimes fails due to network hiccups, test again 5 times to confirm data does not exist
                retry = retry+1
                st_raw_exist = False
                print("Client failed: Retry " + str(retry) + " of 5 attempts")
                import pdb; pdb.set_trace()

        # Check if waveform data exists
        if st_raw_exist == False: #report if no data available 
            print("WARNING: no data available from input station/times")
            utcstart_chunk = utcstart_chunk+DAY_LENGTH
            utcend_chunk = utcend_chunk+DAY_LENGTH
            continue

        try:
            # Remove sensitivity and response, and filter data
            st_raw.detrend(type="linear")
            st_raw.remove_response(output='VEL', pre_filt=[1, 3, 45, 49])
            # st_raw.remove_sensitivity()
        except: #response attachment can also fail due to network hiccups, take a break and try again in 5 minutes.
            # if fails, waits to reset connection then tries again
            print('Connection reset error, retrying')
            time.sleep(60*5)
            continue

        num_sta = len(st_raw)  # count stations in trace
        analyzers_chunk = []  # initiate chunk dataframe
        # Run detector on each station
        for idx in range(1, num_sta+1):

            j = idx - 1
            tr = st_raw[j]

            tr_filt = tr.copy()
            # skip if less than 1 min of data
            if len(tr_filt.data) < tr_filt.stats.sampling_rate*59:
                continue
            # skip if data is bad (indicated by constant data)
            if tr_filt.data[0] == tr_filt.data[1]:
                continue

         
            

            # Build detection metrics 

            # Spectrogram metrics
            window_size = .8
            overlap = .95
            freqlim = [10, 40]
            # SNR metrics
            snr_limits = [(F0+F1)/2-2, (F0+F1)/2+2]
            snr_calllength = 1
            snr_freqwidth = 3
            # Event metrics
            prominence = 500  # 4 #.2 #.5 #min threshold   .1 for 0.3 second window
            event_dur = .1  # minimum width of detection
            distance = 13  # minimum distance between detections
            rel_height = .5 #measures duration at %50 peak height



            # Make spectrogram
            [f, t, Sxx] = detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=window_size,
                                        overlap=overlap, plotflag=PLOTFLAG, filt_freqlim=freqlim, ylim=freqlim)

            # Make detection kernel
            [tvec, fvec, BlueKernel, freq_inds] = detect.buildkernel(
                f0, f1, bdwdth, dur, f, t, tr_filt.stats.sampling_rate, plotflag=PLOTFLAG, kernel_lims=detect.finKernelLims)

            # subset spectrogram to be in same frequency range as kernel
            Sxx_sub = Sxx[freq_inds, :][0]
            f_sub = f[freq_inds]

            # Run detection using built kernel and spectrogram
            #[times, values]=detect.xcorr_log(t,f_sub,Sxx_sub,tvec,fvec,BlueKernel, plotflag=PLOTFLAG,ylim=freqlim)

            [times, values] = detect.xcorr(
                t, f_sub, Sxx_sub, tvec, fvec, BlueKernel, plotflag=PLOTFLAG, ylim=freqlim)

            # Pick detections using EventAnalyzer class
            analyzer_j = EventAnalyzer(times, values, utcstart_chunk - .5*CHUNK_LENGTH, dur=event_dur,
                                    prominence=prominence, distance=distance, rel_height=rel_height)
            # analyzer_j.plot()

            #import pdb; pdb.set_trace()

            
            #[snr,ambient_snr,db_amps] = detect.get_snr(analyzer_j, t, f_sub, Sxx_sub, utcstart_chunk  - .5*CHUNK_LENGTH,snr_limits=snr_limits,
            #                            snr_calllength=snr_calllength,snr_freqwidth=snr_freqwidth,dur=dur,fs=tr_filt.stats.sampling_rate,window_len=window_size)
            #import pdb; pdb.set_trace()

            if len(analyzer_j.df) >= 1:
            #mp_df_time = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                samples = list(range(0, len(tr_filt.data)))
                # Design bandpass filter to look between SNR limits of call (make these wide enough for both call types)
                sos = sig.butter(4, np.array(snr_limits), 'bp',
                                fs=tr_filt.stats.sampling_rate, output='sos')
                # filter timeseries with bandpass filter
                filtered_data = sig.sosfiltfilt(sos, tr_filt.data)
                # take hilbert envelope of timeseries
                amplitude_envelope = abs(hilbert(filtered_data))
                # calculate seconds
                seconds = np.array(
                    [s/tr_filt.stats.sampling_rate for s in samples])

                [maxamp, ambient_snr, snr, medamp] = detect.amps_snr_timeseries(
                    seconds, amplitude_envelope, utcstart_chunk  - .5*CHUNK_LENGTH, analyzer_j, 3, 2)  # get amplitudes in array

                sos_eq = sig.butter(4, np.array([10,14]), 'bp',
                                fs=tr_filt.stats.sampling_rate, output='sos')
                # filter timeseries with bandpass filter
                filtered_data_eq = sig.sosfiltfilt(sos_eq, tr_filt.data)
                # take hilbert envelope of timeseries
                amplitude_envelope_eq = abs(hilbert(filtered_data_eq))
                # calculate seconds

                [maxamp_eq, ambient_snr_eq, snr_eq, medamp_eq] = detect.amps_snr_timeseries(
                    seconds, amplitude_envelope_eq, utcstart_chunk  - .5*CHUNK_LENGTH, analyzer_j, 3, 2)  # get amplitudes in array
                #import pdb; pdb.set_trace()
            else:
                snr=[]
                #import pdb; pdb.set_trace()
                continue


            # Make dataframe with detections from current time chunk
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
            analyzer_j.df['snr_eq'] = snr_eq
            analyzer_j.df['ambient_snr_eq'] = ambient_snr_eq
            analyzer_j.df['db_amps_eq'] = 20*np.log10(maxamp_eq)
            analyzer_j.df[cn.STATION_CODE] = station_codes
            analyzer_j.df[cn.NETWORK_CODE] = network_codes
            analyzer_j.df['quality'] = np.repeat(
                0, analyzer_j.df.shape[0])
            analyzer_j.df['peak_epoch'] = peak_epoch
            analyzer_j.df['start_epoch'] = start_epoch
            analyzer_j.df['end_epoch'] = end_epoch
            analyzers_chunk.append(analyzer_j.df)

            # begin multipath method
            data_starttime = tr_filt.stats.starttime
            utc_times = [data_starttime + ti for ti in times]
            timescount = int(round(CHUNK_LENGTH/(utc_times[1]-utc_times[0])))
            one_minute = int(60/(utc_times[1]-utc_times[0]))

            # find number of minutes for range calculation
            #numchunks = round((utcend_chunk-utcstart_chunk)/60)
            numchunks = round(((tr_filt.stats.endtime-tr_filt.stats.starttime)-CHUNK_LENGTH)/60)
            
            
            if MP_TS_FLAG == True:
                for mp_iter in range(numchunks): #iterate for each minute in day
                    print(mp_iter)
                    startind = one_minute*mp_iter
                    endind = one_minute*mp_iter+timescount

                    times_sub = times[startind:endind]
                    values_sub = values[startind:endind]

                    #import pdb; pdb.set_trace()
                    start_utc=utc_times[startind]
                    end_utc=utc_times[endind]

                    j_df_sub=analyzer_j.df.loc[(analyzer_j.df['peak_time'] >= start_utc) & 
                            (analyzer_j.df['peak_time'] < end_utc) & (analyzer_j.df['db_amps']+5 > analyzer_j.df['db_amps_eq'])]

                    min_df_sub=analyzer_j.df.loc[(analyzer_j.df['peak_time'] >= (start_utc+CHUNK_LENGTH/2-30)) & 
                            (analyzer_j.df['peak_time'] < (end_utc-CHUNK_LENGTH/2+30)) & (analyzer_j.df['db_amps']+5 > analyzer_j.df['db_amps_eq'])]

                    #dt_up = .6  # 7
                    #dt_down = 8.2

                    #import pdb; pdb.set_trace()
                    #Only attempt ranging if there are at least 2 calls in center minute and 5 calls in surrounding 10 minutes
                    if len(min_df_sub) > 1 and len(j_df_sub) >= 5:  #Multipath ranging process through autocorrelation
                        #import pdb; pdb.set_trace()
                        quality_inds = j_df_sub.index.tolist()
                        analyzer_j.df['quality'][quality_inds] = 1
                        #import pdb; pdb.set_trace()
                        # detection score autocorrelation
                        det_timesnew = np.linspace(
                            min(times_sub), max(times_sub), len(times_sub)*10)
                        from scipy.interpolate import interp1d
                        from scipy import signal
                        #import pdb; pdb.set_trace()
                        f = interp1d(times_sub, values_sub, kind='cubic')
                        det_valuesnew1 = f(det_timesnew)
                        det_valuesnew = sig.detrend(det_valuesnew1, type='constant')
                        corr = signal.correlate(det_valuesnew, det_valuesnew)
                        # autocorr_chunk=corr[len(det_timesnew)-1:]/max(corr)
                        autocorr_chunk = corr[len(det_timesnew)-math.ceil(dt_up/(det_timesnew[1]-det_timesnew[0])):len(
                            det_timesnew)+math.ceil(dt_down/(det_timesnew[1]-det_timesnew[0]))]/max(corr)
                        dettimes_chunk = det_timesnew[0:len(autocorr_chunk)]


                        # calculate multipaths for autocorrelated detection score
                        mp_df_auto = pd.DataFrame(columns=cn.SCM_MULTIPATHS.columns)
                        dettimes_chunk = dettimes_chunk-min(dettimes_chunk)
                        mp_event = analyzer_j.mp_picker(
                                dettimes_chunk, autocorr_chunk, start_utc, dur=.1, prominence=0, distance=1, rel_height=.5)
                        for det in range(0, len(min_df_sub)):
                            
                            mp_df_auto = mp_df_auto.append(mp_event, ignore_index=True)
                        
                        min_df_sub = pd.concat([min_df_sub, mp_df_auto], axis=1)
                        d2 = {'date': [start_utc+CHUNK_LENGTH/2], 'epoch': datetimeToEpoch([start_utc+CHUNK_LENGTH/2]), 'n_calls': [len(mp_df_auto)], 'sum_calls': [len(j_df_sub)], 'peaks': [
                            np.median(j_df_sub['peak_signal'])], 'snr': [np.median(j_df_sub['snr'])], 'db_amps': [np.median(j_df_sub['db_amps'])]}
                        auto_df = pd.DataFrame(d2)
                        auto_df = pd.concat([auto_df, mp_df_auto.head(1)], axis=1)
                        #import pdb; pdb.set_trace()
                        # make stack and auto dataframes
                        auto_df_full.append(auto_df)
                        # stack_df_full.append(stack_df)
                        #import pdb; pdb.set_trace()
                        
                        # new_stack_df=pd.concat(stack_df_full)
                        
                        #new_stack_df.to_csv('stack_Marianas_B19_v2.csv', index=False)


                        # Build array for peak displays
                        #timeinds=np.where((stacktimes > mp_event['arrival_1'][0]) & (stacktimes < mp_event['arrival_1'][0] + 9))
                        #timeinds = np.where((dettimes_chunk >= mp_event['arrival_1'][0]) & (
                        #   dettimes_chunk <= mp_event['arrival_1'][0] + 9))
                        #import pdb; pdb.set_trace()
                        # y_values=stackvalues[timeinds]
                        #y_values = autocorr_chunk[timeinds]
                        #ar = np.append(ar, [y_values], axis=0)
                        # times=stacktimes[timeinds]
                        #times = dettimes_chunk[timeinds]
                        #yax = times-min(times)
                    #else:
                        #y_values = np.zeros((1, 205))
                        # y_values=np.zeros((1,2251))
                        #ar = np.append(ar, y_values, axis=0)

                    #import pdb; pdb.set_trace()
                    # plots stacked spectrogram
                    if PLOTFLAG_STACK and len(analyzer_j.df) > 1:

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

        
        utcstart_chunk = utcstart_chunk+DAY_LENGTH
        utcend_chunk = utcend_chunk+DAY_LENGTH

        # Extend final dataframe with detections from current time chunk
        analyzers.extend(analyzers_chunk)

        #import pdb; pdb.set_trace()
        new_df = pd.concat(analyzers)
        new_df.to_csv(chunk_pth, index=False)
        try:
            new_auto_df = pd.concat(auto_df_full)
            new_auto_df.to_csv('auto_'+CHUNK_FILE, index=False)
        except:
            continue

        

    if len(analyzers) == 0:
        print('WARNING: detections dataframe empty')
        final_analyzer_df = []
    elif len(analyzer_j.df) == 0:
        print('WARNING: no detections from current time window')
        #final_analyzer_df = []
    else:
        
        final_analyzer_df = pd.concat(analyzers)
        final_analyzer_df.to_csv(detection_pth, index=False)
        #import pdb; pdb.set_trace()
        
        
 

sta_table = pd.read_csv('Marianas_sites_example.csv',parse_dates=['startdate','enddate'])



for site_ind in range(0,len(sta_table['Sites'])): 
    network = sta_table['Network'][site_ind]   # Network name "OO" for OOI, "7D" for Cascadia, "XF" for PuertoRico, "XO" for alaska, "YO" for ENAM, "9A" for Hawaii, "ZZ" for puerto rico
    station = sta_table['Sites'][site_ind]    # "LD41" # "B19" for PuertoRico station #Specific station, or '*' for all available stations "X06" for ENAM, "LOSW" for Hawaii, "XABV" for puerto rico
    location = '*'  # '*' for all available locations
    channel = sta_table['Channel'][site_ind]   # Choose channels,  you'll want 'BHZ,HHZ' for Cascadia 'ELZ' for hawaii
    depth = sta_table['Instrument Depth (m)'][site_ind]

    ss=1512
    sed_speed = 1512
    t = sta_table['Reflector depth (m)'][site_ind] - depth

    [distance,t0,t1,t2,t3,t0_1,t1_1,t1_2]=ranging.basic_ranging(depth,ss,sed_speed,t,plotflag=False)
    max_diff=max(np.subtract(t1_1,t0))
    dt_down = max_diff + 0.5

    CHUNK_FILE = station+'_mp_CORTADO_TEST.csv'  
    DET_PATH = CHUNK_FILE
    STARTTIME = sta_table['startdate'][site_ind]
    ENDTIME = sta_table['enddate'][site_ind]

    

    main(STARTTIME, ENDTIME,
        client_code=CLIENT_CODE, f0=F0,
        f1=F1, bdwdth=BDWDTH, dur=DUR,
        detection_pth=DET_PATH,
        chunk_pth=CHUNK_FILE, station_ids=station,
        is_restart=True, dt_up = dt_up, dt_down = dt_down)





#if __name__ == "__main__":
#    main(STARTTIME, ENDTIME)

#________  ________  ________  _________  ________  ________  ________     
#|\   ____\|\   __  \|\   __  \|\___   ___\\   __  \|\   ___ \|\   __  \    
#\ \  \___|\ \  \|\  \ \  \|\  \|___ \  \_\ \  \|\  \ \  \_|\ \ \  \|\  \   
# \ \  \    \ \  \\\  \ \   _  _\   \ \  \ \ \   __  \ \  \ \\ \ \  \\\  \  
#  \ \  \____\ \  \\\  \ \  \\  \|   \ \  \ \ \  \ \  \ \  \_\\ \ \  \\\  \ 
#   \ \_______\ \_______\ \__\\ _\    \ \__\ \ \__\ \__\ \_______\ \_______\
#    \|_______|\|_______|\|__|\|__|    \|__|  \|__|\|__|\|_______|\|_______|
