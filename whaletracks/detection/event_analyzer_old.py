#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:15:18 2020

@author: wader
"""

import whaletracks.common.constants as cn
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from whaletracks.common import util
import math

#b-call dur=5, rel_height=.7 and prominence=.5 wlen=60 seconds
#a-call dur=70, rel_height=.9 prominence=.3 wlen=2 min distance=30
#fin-call dur=.5, rel_height=.3 prominence=.6 wlen=60 sec distance=18
#REL_HEIGHT=.3
SECONDS_IN_MINUTE = 60
EXCLUDED_COLUMNS = [cn.THRESHOLD, cn.STATION_CODE, cn.NETWORK_CODE]

class EventAnalyzer(object):
    
    def __init__(self, times, values, start_chunk, dur=1, prominence=.6, distance=18, rel_height=.8):
        """
        :param list-float times: offsets in seconds
        :param list-float values: values at times
        :param UTCdatetime: start time of chunk
        :param float dur: duration in seconds of call (default for blue whale)
        """
        self.times = [start_chunk + t for t in times]
        self.values = values
        self.start_chunk = start_chunk
        peak_indicies, peak_properties=sig.find_peaks(self.values,
            distance=distance*(1/(self.times[1]-self.times[0])),
            width=dur*(1/(self.times[1]-self.times[0])),
            prominence=prominence,
            wlen=SECONDS_IN_MINUTE*(1/(self.times[1]-self.times[0])),
            rel_height=rel_height)
        
        self.df = self._makeDetectionDF(peak_indicies, peak_properties)
        self.df[cn.THRESHOLD] = prominence
        
     
    def _makeDetectionDF(self, peak_indicies, peak_properties):
        """
        :param int index: index of peak
        :return pd.DataFrame: all columns, except for EXCLUDED_COLUMNS
        """
        dct = {k: [] for k in cn.SCM_DETECTION.columns 
               if not k in EXCLUDED_COLUMNS}
        for index in range(len(peak_indicies)):
            dct[cn.PEAK_TIME].append(self.times[peak_indicies[index]])
            dct[cn.PEAK_SIGNAL].append(peak_properties["prominences"][index])
            dct[cn.START_TIME].append(
                    self.times[peak_properties["left_ips"].astype(int)[index]])
            dct[cn.END_TIME].append(
                    self.times[peak_properties["right_ips"].astype(int)[index]])
            dct[cn.MIN_SIGNAL].append(peak_properties["width_heights"][index])
            
            dct[cn.DURATION].append(self.times[peak_properties["right_ips"].astype(int)[index]]-
                                 self.times[peak_properties["left_ips"].astype(int)[index]])
            dct[cn.PEAK_EPOCH] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_EPOCH] = list(np.repeat(None, len(dct[cn.START_TIME])))
            dct[cn.END_EPOCH] = list(np.repeat(None, len(dct[cn.END_TIME])))
            dct[cn.SNR] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.SNR_AMBIENT] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.PEAK_FREQUECNY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_FREQUENCY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.END_FREQUENCY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.PEAK_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.END_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
        #import pdb; pdb.set_trace()
            

        return pd.DataFrame(dct)

    
    def mp_picker(self, times, values, utcstart_chunk, dur=.5, prominence=.1):
        """
        :param list-float times: offsets in seconds
        :param list-float values: values at times
        :param UTCdatetime: start time of chunk
        :param float dur: duration in seconds of call (default for blue whale)
        """
        
        peak_indicies, peak_properties=sig.find_peaks(values,
            distance=.1*(1/(times[1]-times[0])),
            width=(dur/2)*(1/(times[1]-times[0])),
            prominence=prominence,
            wlen=SECONDS_IN_MINUTE*(1/(times[1]-times[0])),
            rel_height=.2)

        #import pdb; pdb.set_trace();
        for ind in range(len(peak_properties['prominences'])):
            peak_properties['prominences'][ind]=peak_properties['prominences'][ind]+max([values[peak_properties['left_bases'][ind]],values[peak_properties['right_bases'][ind]]])

        #import pdb; pdb.set_trace();  
        mp_df_j = self.makeMultipathDF(peak_indicies, peak_properties, times, values, utcstart_chunk)

        return mp_df_j

    def makeMultipathDF(self, peak_indicies, peak_properties, times, values, utcstart_chunk):
        """
        :param int index: index of peak
        :return pd.DataFrame: all columns, except for EXCLUDED_COLUMNS
        """
        dct = {k: [] for k in cn.SCM_DETECTION.columns 
               if not k in EXCLUDED_COLUMNS}
        for index in range(len(peak_indicies)):
            dct[cn.PEAK_TIME].append(times[peak_indicies[index]])
            dct[cn.PEAK_SIGNAL].append(peak_properties["prominences"][index])
            dct[cn.START_TIME].append(
                    times[peak_properties["left_ips"].astype(int)[index]])
            dct[cn.END_TIME].append(
                    times[peak_properties["right_ips"].astype(int)[index]])
            dct[cn.MIN_SIGNAL].append(peak_properties["width_heights"][index])
            
            dct[cn.DURATION].append(times[peak_properties["right_ips"].astype(int)[index]]-
                                 times[peak_properties["left_ips"].astype(int)[index]])
            dct[cn.PEAK_EPOCH] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_EPOCH] = list(np.repeat(None, len(dct[cn.START_TIME])))
            dct[cn.END_EPOCH] = list(np.repeat(None, len(dct[cn.END_TIME])))
            dct[cn.SNR] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.SNR_AMBIENT] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.PEAK_FREQUECNY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_FREQUENCY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.END_FREQUENCY] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.PEAK_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.START_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))
            dct[cn.END_FREQUENCY_STD] = list(np.repeat(None, len(dct[cn.PEAK_TIME])))

        event_df=pd.DataFrame(dct)
        event_peaksort=event_df.sort_values(by=['peak_signal'],ascending=False)[0:5]
        event_timesort=event_peaksort.sort_values(by=['peak_time'])
        arrivals_sec = event_timesort['peak_time'].values.tolist()
        #arrivals = [utcstart_chunk + a for a in arrivals_sec]
        arrivals=arrivals_sec
        amplitudes = event_timesort['peak_signal'].values.tolist()
        nonelen=5-len(arrivals)
        nonelist=list(np.repeat(None, nonelen))
        arrivals=arrivals+nonelist
        amplitudes=amplitudes+nonelist
        
        mp_dct = {k: [] for k in cn.SCM_MULTIPATHS.columns 
               if not k in EXCLUDED_COLUMNS}
        #import pdb; pdb.set_trace()
        mp_dct[cn.ARRIVAL_1].append(arrivals[0])
        mp_dct[cn.ARRIVAL_2].append(arrivals[1])
        mp_dct[cn.ARRIVAL_3].append(arrivals[2])
        mp_dct[cn.ARRIVAL_4].append(arrivals[3])
        mp_dct[cn.ARRIVAL_5].append(arrivals[4])
        mp_dct[cn.AMP_1].append(amplitudes[0])
        mp_dct[cn.AMP_2].append(amplitudes[1])
        mp_dct[cn.AMP_3].append(amplitudes[2])
        mp_dct[cn.AMP_4].append(amplitudes[3])
        mp_dct[cn.AMP_5].append(amplitudes[4])
        mp_df = pd.DataFrame(mp_dct)
        
        return mp_df
       

        
    def plot(self, is_plot=True):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(self.times,self.values)
        ax.plot(self.df.peak_time,self.df.peak_signal,'x')
        plt.hlines(self.df.min_signal,self.df.start_time,
                   self.df.end_time,color="C2")
        if is_plot:
            plt.show(block=True)

    
