from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import scipy.io.wavfile as siow
import numpy as np
import scipy.signal as sig
import matplotlib.colors as color
import matplotlib.animation as animation
from datetime import datetime
import math
import numpy.matlib



def datetimeToEpoch(UTCdatetime_list):
    """
    Converts list of datestrings in UTCDateTime format
    into epoch time (seconds elapsed since 01-01-1970)
    :param list datestrs: list of datestrings
    :param str dateformat: date format as accepted by datetime.strptime
    :return list floats: seconds elapsed since 01-01-1970
    """
    #import pdb; pdb.set_trace()
    epochlist=list()
    SECONDS_IN_DAY=86400
    datetime_epochstart = datetime.strptime('1970-01-01T00:00:00.00Z',
                                            '%Y-%m-%dT%H:%M:%S.%fZ')

    #import pdb; pdb.set_trace()
    for k in range(1,len(UTCdatetime_list)+1):
        j=k-1
        datetime_j = UTCdatetime_list[j]
        epoch_delta = datetime_j.datetime - datetime_epochstart
        epoch_j = epoch_delta.days*SECONDS_IN_DAY + epoch_delta.seconds + epoch_delta.microseconds/1000000
        
        epochlist.append(epoch_j)
        
    return epochlist

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_snr(picks,t,f,Sxx,utcstart_chunk,snr_limits=[12, 16],snr_calllength=4,snr_freqwidth=.6,dur=10):

    peak_times=picks
    snr=[]
    #import pdb; pdb.set_trace()
    freq_inds=np.where(np.logical_and(f>=min(snr_limits), f<=max(snr_limits)))
    Sxx_sub=Sxx[freq_inds,:][0]
    f=f[freq_inds]

    med_noise = np.median(Sxx_sub)
    utc_t = [utcstart_chunk + j for j in t]   
    snr_t_int=np.int((snr_calllength)/(utc_t[1] - utc_t[0]))
    for utc_time in peak_times:

        #import pdb; pdb.set_trace()
        t_peak_ind = find_nearest(utc_t,utc_time)
        #t_peak_ind = utc_t.index(utc_time) 
        Sxx_t_inds1 = list(range(t_peak_ind,t_peak_ind+snr_t_int))
        #import pdb; pdb.set_trace()
        Sxx_t_inds = [x for x in Sxx_t_inds1 if x < len(t)]
        Sxx_t_sub = Sxx_sub[:,Sxx_t_inds]
        db_max = np.max(Sxx_sub[:,t_peak_ind])
        max_loc = np.where(Sxx_sub[:,t_peak_ind] == db_max)
        freq_max=f[max_loc]
        f_inds = np.where(np.logical_and(f>=freq_max-snr_freqwidth/2, f<=freq_max+snr_freqwidth/2))
        Sxx_tf_sub = Sxx_t_sub[f_inds,:]
        call_noise = np.median(Sxx_tf_sub)
        snr = snr + [10*np.log10(call_noise/med_noise)]

    #Get SNR of 10 seconds of noise preceding call
    start_times=picks
    noise_t_int=np.int((dur)/(utc_t[1] - utc_t[0]))
    start_snr=[]
    for utc_time in start_times:
        
        t_peak_ind = find_nearest(utc_t,utc_time)
        Sxx_t_inds1 = list(range(t_peak_ind-noise_t_int,t_peak_ind))
        #import pdb; pdb.set_trace()
        Sxx_t_inds = [x for x in Sxx_t_inds1 if x >= 0]
        Sxx_t_sub = Sxx_sub[:,Sxx_t_inds]
        ambient_noise = np.median(Sxx_t_sub)
        start_snr = start_snr + [10*np.log10(ambient_noise/med_noise)]

   
    ambient_snr=start_snr
    #ambient_snr=np.divide(np.sum(start_snr,end_snr),2)
    
    return snr, ambient_snr

def defaultScaleFunction(Sxx):
    vmin=np.median(10*np.log10(Sxx))+0*np.std(10*np.log10(Sxx)) 
    vmax=np.median(10*np.log10(Sxx))+2*np.std(10*np.log10(Sxx)) 
    return vmin, vmax


def plotwav(samp, data, filt_type='bandpass', filt_freqlim=[12, 18], 
            filt_order=4, window_size=4, overlap=.95, window_type='hann',
            plotflag=True, scale_func=defaultScaleFunction,ylim=[12, 18]): 
    """
    Calculate spectogram and plot.
    :param float samp: sampling rate
    :param numpy.array data: data to process
    :param string filt_type: filter type (highpass,lowpass,bandpass etc.)
    :param tuple filt_freqlim: frequency limits of filter
    :param int filt_order: order of filter 
    :param float window_size: seconds spectrogram window size
    :param float overlap: ratio of overlap of spectrogram window
    :param string window_type: spectrogram window type
    :param bool plotflag: If True, makes plots. If False, no plot.
    :param function scal_func: Single argument Sxx, returns tuple vmin and vmax floats
    :param tuple ylim: lower and uppet bounds of frequency for spectrogram plot
    :param numpy.array data: vector of min and max spectrogram frequency 
    :return numpy.array, numpy.array, 2-d numpy.array: 
        vector of frequency, vector of seconds, matrix of power
    Key variables
      f - frequency
      t - time (seconds)
      Sxx - Spectrogram of amplitudes
    """

    PLT_TIMESERIES = 1
    FIGSIZE = [9, 3]
    FILTER_OFFSET = 10

    #filter data to spectral bands where B-call is
    [b, a] = sig.butter(filt_order, np.array(filt_freqlim)/samp, filt_type, 'ba') 
    filtered_data = sig.filtfilt(b, a, data)

    
    datalength = data.size
    times = (np.arange(datalength)/samp) 

    #plot timeseries on upper axis
    if plotflag == True:
        plt.figure(PLT_TIMESERIES, figsize=FIGSIZE)
        plt.subplot(211)
        plt.plot(times[FILTER_OFFSET:],filtered_data[FILTER_OFFSET:])
        plt.axis([min(times), max(times), min(filtered_data[FILTER_OFFSET:]), 
                  max(filtered_data[FILTER_OFFSET:])])
        plt.xlabel('Seconds')
        plt.ylabel('Amplitude')

    #plot spectrogram on lower axis
    [f, t, Sxx] = sig.spectrogram(filtered_data, int(samp), 
    window_type,int(samp*window_size),int(samp*window_size*overlap))
    
    if plotflag == True:
        cmap = plt.get_cmap('magma')
        vmin, vmax = scale_func(Sxx)
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        plt.subplot(212)
        plt.pcolormesh(t, f, 10*np.log10(Sxx), cmap=cmap, norm=norm)    
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim(ylim)
        plt.show(PLT_TIMESERIES)
        #plt.clf()
        
    return [f, t, Sxx]

def freq_analysis(picks_t,calltype,t,f,Sxx,utcstart_chunk,freq_window_b=[13, 16.5],freq_window_a=[12, 15.5]):
    peak_freqs = []
    peak_stds = []
 
    #import pdb; pdb.set_trace()
    freq_inds = np.where(np.logical_and(f>=min(freq_window_b), f<=max(freq_window_b)))
    Sxx_sub = Sxx[freq_inds,:][0]
    f_b = f[freq_inds]
    utc_t = [utcstart_chunk + j for j in t]   
    utc_array = np.array(utc_t)

    freq_inds_a = np.where(np.logical_and(f>=min(freq_window_a), f<=max(freq_window_a)))
    Sxx_sub_a = Sxx[freq_inds_a,:][0]
    f_a = f[freq_inds_a]
    utc_t_a = [utcstart_chunk + j for j in t]   
    utc_array_a = np.array(utc_t_a)

    for k in range(0,len(picks_t)): #for peak freq
        if calltype[k] == "B":
            starttime = picks_t[k]+1
            endtime = picks_t[k]+5
            callbool = (utc_array < endtime) & (utc_array > starttime)
            inds = np.array(list(range(len(callbool))))
            callinds = inds[callbool]
            call_times = utc_array[callbool]
            Sxx_total_sub = Sxx_sub[:,callinds]
            farray = np.matlib.repmat(f_b,len(call_times),1)
            peak_freq = np.sum(np.multiply(farray.T,Sxx_total_sub))/np.sum(Sxx_total_sub)
            peak_freqs = peak_freqs + [peak_freq]
            peak_std = np.sum(np.multiply(np.power(np.subtract(farray.T,np.mean(farray)),2),Sxx_total_sub))/np.sum(Sxx_total_sub)
            peak_stds = peak_stds + [peak_std]

        if calltype[k] == "A":
            starttime = picks_t[k]+1
            endtime = picks_t[k]+5
            callbool = (utc_array_a < endtime) & (utc_array_a > starttime)
            inds = np.array(list(range(len(callbool))))
            callinds = inds[callbool]
            call_times = utc_array_a[callbool]
            Sxx_total_sub = Sxx_sub_a[:,callinds]
            farray = np.matlib.repmat(f_a,len(call_times),1)
            peak_freq = np.sum(np.multiply(farray.T,Sxx_total_sub))/np.sum(Sxx_total_sub)
            peak_freqs = peak_freqs + [peak_freq]
            peak_std = np.sum(np.multiply(np.power(np.subtract(farray.T,np.mean(farray)),2),Sxx_total_sub))/np.sum(Sxx_total_sub)
            peak_stds = peak_stds + [peak_std]
        

    return peak_freqs, peak_stds