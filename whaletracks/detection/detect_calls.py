#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:48:49 2020

@author: wader
"""

#Needs inputs of data (just timeseries) sample rate
#The spectrogram work is done by scipy.signal.spectrogram
#Check out the man pages and figure out what kind of windows and overlap you'd like for the best looking spectrogram.

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
import datetime
import math
import numpy.matlib
# Constants
PLT_TIMESERIES = 1
PLT_SPECTROGRAM = 2
PLT_KERNEL = 3
PLT_SCORE = 4
FIGSIZE = [9, 3]

FILTER_OFFSET = 10

# Helper functions

def defaultScaleFunction(Sxx):
    vmin=np.median(10*np.log10(Sxx))+0*np.std(10*np.log10(Sxx)) 
    vmax=np.median(10*np.log10(Sxx))+2*np.std(10*np.log10(Sxx)) 
    return vmin, vmax

def defaultKernelLims(f0,f1,bdwdth):
    ker_1=f1-4*bdwdth 
    ker_2=f0+4*bdwdth
    ker_min=np.min([ker_1,ker_2])
    ker_max=np.max([ker_1,ker_2])
    return ker_min, ker_max

 
def finKernelLims(f0,f1,bdwdth):
    ker_1=0
    ker_2=50
    ker_min=np.min([ker_1,ker_2])
    ker_max=np.max([ker_1,ker_2])
    return ker_min, ker_max  

def plotwav(samp, data, filt_type='bandpass', filt_freqlim=[12, 18], 
            filt_order=2, window_size=4, overlap=.95, window_type='hann',
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
    #filter data to spectral bands where B-call is
    
    sos = sig.butter(filt_order, filt_freqlim, 'bp', fs=samp, output = 'sos') 
    filtered_data = sig.sosfiltfilt(sos, data)

    #sos = sig.butter(filt_order, [24, 26], btype='bandstop', fs=samp, output = 'sos') 
    #filtered_data = sig.sosfiltfilt(sos, data)

    #[b, a] = sig.butter(filt_order, np.array(filt_freqlim)/samp, filt_type, 'ba') 
    #filtered_data = sig.filtfilt(b, a, data)

    
    datalength = data.size
    times = (np.arange(datalength)/samp) 

    [f, t, Sxx] = sig.spectrogram(filtered_data, samp, 
    window_type,int(samp*window_size),int(samp*window_size*overlap),return_onesided=True)

    #plot timeseries on upper axis
    if plotflag == True:
        plt.figure(PLT_TIMESERIES, figsize=FIGSIZE)
        fig, (ax0, ax1) = plt.subplots(nrows=2,sharex=True)
        #plt.subplot(211)
        ax0.plot(times[FILTER_OFFSET:],filtered_data[FILTER_OFFSET:])
        #plt.axis([min(times), max(times), min(filtered_data[FILTER_OFFSET:]), 
                  #max(filtered_data[FILTER_OFFSET:])])
        ax0.set_xlabel('Seconds')
        ax0.set_ylabel('Amplitude')
        ax0.set_title('Filtered timeseries and spectrogram of test data')

        cmap = plt.get_cmap('magma')
        vmin, vmax = scale_func(Sxx)
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        #plt.subplot(212)
        ax1.pcolormesh(t, f, 10*np.log10(Sxx), cmap=cmap, norm=norm)    
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylim(ylim)
        plt.show()
        #plt.clf()
        
    return [f, t, Sxx]


#Define function to build 2-D kernel linear sweep to cross-correlate with spectrograms
def buildkernel(f0, f1, bdwdth, dur, f, t, samp, plotflag=True,kernel_lims=finKernelLims):
    
    """
    Calculate kernel and plot
    :param float f0: starting frequency
    :param float f1: ending frequency
    :param float bdwidth: frequency width of call
    :param float dur: call length (seconds)
    :param np.array f: vector of frequencies returned from plotwav
    :param np.array t: vector of times returned from plotwav
    :param float samp: sample rate
    :param bool plotflag: If True, plots kernel. If False, no plot.
    :param tuple kernel_lims: Tuple of minimum kernel range and maximum kernel range
    :return numpy.array, numpy.array, 2-d numpy.array: 
        vector of kernel times, vector of kernel frequencies, matrix of kernel values
    Key variables
      tvec - kernel times (seconds)
      fvec - kernel frequencies
      BlueKernel - Matrix of kernel values
    """
    
    tvec = np.linspace(0,dur,np.size(np.nonzero((t < dur*8) & (t > dur*7)))) #define kernel as length dur
    fvec = f #define frequency span of kernel to match spectrogram
    Kdist = np.zeros((np.size(tvec), np.size(fvec))) #preallocate space for kernel values
    ker_min, ker_max=kernel_lims(f0,f1,bdwdth)
    #import pdb; pdb.set_trace()
    for j in range(np.size(tvec)):
        #calculate hat function that is centered on linearly decresing
        #frequency values for each time in tvec
        x = fvec-(f0+(tvec[j]/dur)*(f1-f0))
        Kval = (1-np.square(x)/(bdwdth*bdwdth))*np.exp(-np.square(x)/(2*(bdwdth*bdwdth)))
        Kdist[j] = Kval #store hat function values in preallocated array

    #import pdb; pdb.set_trace()                                                    
    BlueKernel_full = np.transpose(Kdist) #transpose preallocated array to be plotted vs. tvec and fvec
    freq_inds=np.where(np.logical_and(fvec>=ker_min, fvec<=ker_max))
    
    fvec_sub=fvec[freq_inds]
    BlueKernel=BlueKernel_full[freq_inds,:][0]
    
    #import pdb; pdb.set_trace();
    if plotflag == True:
        plt.figure(PLT_KERNEL)
        cmap = plt.get_cmap('bone')
        plt.pcolormesh(tvec, fvec_sub, BlueKernel,cmap=cmap) #show plot of kernel
        plt.axis([0, dur, np.min(fvec), np.max(fvec)])
        #plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.ylim(f1-8,f0+8)
        plt.title('Blue whale B-call kernel')
        plt.show()
        #plt.clf()
        
    return [tvec, fvec_sub, BlueKernel, freq_inds]




def xcorr(t,f,Sxx,tvec,fvec,BlueKernel,plotflag=True,scale_func=defaultScaleFunction,ylim=[12, 18]):
    """
    Cross-correlate kernel with spectrogram and plot score
    :param np.array f: vector of frequencies returned from plotwav
    :param np.array t: vector of times returned from plotwav
    :param np.array Sxx: 2-D array of spectrogram amplitudes
    :param np.array tvec: vector of times of kernel
    :param np.array fvec: vector of frequencies of kernel
    :param np.array BlueKernel: 2-D array of kernel amplitudes
    :plotflag boolean plotflag: Boolean. If True, plots result. If False, no plot.
    :param function scal_func: Single argument Sxx, returns tuple vmin and vmax floats
    :return numpy.array, numpy.array:
        vector of correlation times, vector of correlation values
    Key variables
        t_scale - correlation times (seconds)
        CorrVal_scale - correlation values
    """

    #plt.rcParams.update({'font.size': 10})
    Sxx_log1=10*np.log10(Sxx)
    #Sxx_log1=Sxx_log1-np.min(Sxx_log1)

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
#Visualize spectrogram and detection scores of of data  
    #import pdb; pdb.set_trace()
    if plotflag==True:
        
        t1=min(t)
        t2=max(t)
#plot timeseries on upper axis
        plt.figure(PLT_SCORE, figsize=(9, 3))
        fig, (ax0, ax1) = plt.subplots(nrows=2,sharex=True)
        ax0.plot(t_scale,CorrVal_scale) #plot normalized detection scores as a time series.
        
        ax0.set_xlim([t1, t2]) #look at only positive values
        ax0.set_ylim([0, np.max(CorrVal_scale)])
        ax0.set_xlabel('Seconds')
        ax0.set_ylabel('Detection score')
        ax0.set_title('Spectrogram and detection scores of test data')

#plot spectrogram on lower axis
        cmap = plt.get_cmap('gist_heat')
        vmin=np.median(Sxx_log)+2*np.std(Sxx_log)
        vmax=np.median(Sxx_log)
        vmin1=np.median(Sxx_log1)+2*np.std(Sxx_log1)
        vmax1=np.median(Sxx_log1)
        #vmin, vmax = scale_func(Sxx_log)
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        norm1 = color.Normalize(vmin=vmin1, vmax=vmax1)
        #plt.subplot(212)
        #im = ax1.pcolormesh(t, f, Sxx_log, cmap=cmap,norm=norm) 
        im = ax1.pcolormesh(t, f, Sxx_log1, cmap=cmap,norm=norm1) 
        fig.colorbar(im, ax=ax1,orientation='horizontal')
        ax1.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
        ax1.set_ylim(ylim)
        ax1.set_ylabel('Frequency [Hz]')
        #ax1.set_xticks([])
        #ax1.set_xlabel('Time [seconds]')
        fig.tight_layout()
        plt.show()
        #plt.clf()
    return  [t_scale, CorrVal_scale] 

    #plt.savefig('Spectrogram_scores.png')



def xcorr_log(t,f,Sxx,tvec,fvec,BlueKernel,plotflag=True,scale_func=defaultScaleFunction,ylim=[12, 18]):
    """
    Cross-correlate kernel with spectrogram and plot score
    :param np.array f: vector of frequencies returned from plotwav
    :param np.array t: vector of times returned from plotwav
    :param np.array Sxx: 2-D array of spectrogram amplitudes
    :param np.array tvec: vector of times of kernel
    :param np.array fvec: vector of frequencies of kernel
    :param np.array BlueKernel: 2-D array of kernel amplitudes
    :plotflag boolean plotflag: Boolean. If True, plots result. If False, no plot.
    :param function scal_func: Single argument Sxx, returns tuple vmin and vmax floats
    :return numpy.array, numpy.array:
        vector of correlation times, vector of correlation values
    Key variables
        t_scale - correlation times (seconds)
        CorrVal_scale - correlation values
    """
    Sxx_log1=10*np.log10(Sxx)
    Sxx_log=Sxx_log1-np.min(Sxx_log1)
#Cross-correlate kernel with spectrogram
    ind1 = 0
    CorrVal = np.zeros(np.size(t) - (len(tvec)-1)) #preallocate array for correlation values
    corrchunk= np.zeros((np.size(fvec), np.size(tvec))) #preallocate array for element-wise multiplication

    while ind1-1+np.size(tvec) < np.size(t):
        ind2 = ind1 + np.size(tvec) #indices of spectrogram subset to multiply
        for indF in range(np.size(fvec)-1):
            corrchunk[indF] = Sxx_log[indF][ind1:ind2] #grab spectrogram subset for multiplication
        
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
#Visualize spectrogram and detection scores of of data  
    #import pdb; pdb.set_trace()
    if plotflag==True:
        
        t1=min(t)
        t2=max(t)
#plot timeseries on upper axis
        plt.figure(PLT_SCORE, figsize=(9, 3))
        fig, (ax0, ax1) = plt.subplots(nrows=2,sharex=True)
        ax0.plot(t_scale,CorrVal_scale) #plot normalized detection scores as a time series.
        
        ax0.set_xlim([t1, t2]) #look at only positive values
        ax0.set_ylim([0, np.max(CorrVal_scale)])
        ax0.set_xlabel('Seconds')
        ax0.set_ylabel('Detection score')
        ax0.set_title('Spectrogram and detection scores of test data')

#plot spectrogram on lower axis
        cmap = plt.get_cmap('magma')
        vmin=np.median(Sxx_log)+2*np.std(Sxx_log)
        vmax=np.median(Sxx_log)
        #vmin, vmax = scale_func(Sxx_log)
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        #plt.subplot(212)
        im = ax1.pcolormesh(t, f, Sxx_log, cmap=cmap,norm=norm) 
        fig.colorbar(im, ax=ax1,orientation='horizontal')
        ax1.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
        ax1.set_ylim(ylim)
        ax1.set_ylabel('Frequency [Hz]')
        #ax1.set_xticks([])
        #ax1.set_xlabel('Time [seconds]')
        fig.tight_layout()
        plt.show()
        #plt.clf()
    return  [t_scale, CorrVal_scale] 

    #plt.savefig('Spectrogram_scores.png')

def spect_autocorr(t,f,Sxx,seconds,plotflag=True,scale_func=defaultScaleFunction,ylim=[12, 18]):
    """
    Cross-correlate kernel with spectrogram and plot score
    :param np.array f: vector of frequencies returned from plotwav
    :param np.array t: vector of times returned from plotwav
    :param np.array Sxx: 2-D array of spectrogram amplitudes
    :plotflag boolean plotflag: Boolean. If True, plots result. If False, no plot.
    :param function scal_func: Single argument Sxx, returns tuple vmin and vmax floats
    :return numpy.array, numpy.array:
        vector of correlation times, vector of correlation values
    Key variables
        t_scale - correlation times (seconds)
        CorrVal_scale - correlation values
    """
    Sxx_log1=10*np.log10(Sxx)
    Sxx_log=Sxx_log1-np.min(Sxx_log1)
#Cross-correlate kernel with spectrogram
    ind1 = 0
    autocorr_times=t[t<seconds]
    CorrVal=np.zeros(len(autocorr_times))
    freq_inds=np.where(np.logical_and(f>=min(ylim), f<=max(ylim)))
    fvec=f[freq_inds[0]]
    spec_fsub=Sxx_log[freq_inds[0]][:]

    spec_median=np.median(Sxx_log)
    buffer=spec_median*np.ones((len(freq_inds[0]), len(autocorr_times)))
    spect_buffer=np.concatenate((spec_fsub,buffer),axis=1)

    corrchunk=np.zeros((len(fvec),len(t)))

    #import pdb; pdb.set_trace()
    while ind1 < len(autocorr_times): 

        ind2=len(t)+ind1
        for indF in range(np.size(fvec)-1):
            corrchunk[indF] = spect_buffer[indF][ind1:ind2] #grab spectrogram subset for multiplication
        
        CorrVal[ind1] = np.sum(np.multiply(corrchunk-spec_median, spec_fsub-spec_median)) #save cross-correlation value for each frame
        ind1 += 1
    autodetrend=sig.detrend(CorrVal,type='linear')
    return [autocorr_times, CorrVal]

 
 
def stack_spect(t,f_sub,Sxx,utcstart_chunk,analyzer_j,dt_up,dt_down,pad_length = 2000):

    #Sxx_log1=10*np.log10(Sxx)
    #Sxx_log=Sxx_log1-np.min(Sxx_log1)

    dettimes=analyzer_j.df.peak_time-utcstart_chunk
    samp_down=math.ceil(dt_down/(t[1]-t[0]))
    samp_up=math.ceil(dt_up/(t[1]-t[0]))
    ones_pad=np.ones((len(f_sub),pad_length))*np.median(Sxx)
    Sxx_pad=np.concatenate((ones_pad,Sxx,ones_pad),axis=1)

    stacked_spect=np.zeros((len(f_sub),samp_down+samp_up))
    
    for det in dettimes:
        
        spect_chunk=np.zeros((len(f_sub),samp_down+samp_up))
        minvec=abs(np.subtract(t,det))
        timeind1=np.where(minvec == min(minvec))
        #timeind1=np.where(t == det)
        timeind=timeind1[0][0]+pad_length
        startind=timeind-samp_up
        endind=timeind+samp_down

        for indF in range(np.size(f_sub)):
            
            spect_chunk[indF] = Sxx_pad[indF][startind:endind] #grab spectrogram subset for multiplication

        stacked_spect=np.add(stacked_spect,spect_chunk)
    
    stacked_t=range(len(stacked_spect[0]))*(t[1]-t[0])
    return [stacked_t,f_sub,stacked_spect]


def stack_times(t,env,utcstart_chunk,analyzer_j,dt_up,dt_down,pad_length = 2000):

    #Sxx_log1=10*np.log10(Sxx)
    #Sxx_log=Sxx_log1-np.min(Sxx_log1)

    dettimes=analyzer_j.df.peak_time-utcstart_chunk
    samp_down=math.ceil(dt_down/(t[1]-t[0]))
    samp_up=math.ceil(dt_up/(t[1]-t[0]))
    ones_pad=np.ones((1,pad_length))*np.median(env)
    
    env_pad=np.concatenate((ones_pad[0],env,ones_pad[0]))
    stacked_env=np.zeros((1,samp_down+samp_up))
    
    maxamp=[]

    for det in dettimes:
        
        env_chunk=np.zeros((1,samp_down+samp_up))
        #import pdb; pdb.set_trace()
        timeind1=(np.abs(t - det)).argmin()
        timeind=timeind1+pad_length
        startind=timeind-samp_up
        endind=timeind+samp_down

        env_chunk = env_pad[startind:endind] #grab spectrogram subset for multiplication

        stacked_env=np.add(stacked_env,env_chunk)
        maxamp += [10*np.log10(max(env_chunk))]
    
    stacked_t=np.transpose(range(len(stacked_env[0]))*(t[1]-t[0]))
    return [stacked_t,stacked_env[0],maxamp]

def amps_snr_timeseries(t,env,utcstart_chunk,analyzer_j,dt_up,dt_down,pad_length = 2000):

    #Sxx_log1=10*np.log10(Sxx)
    #Sxx_log=Sxx_log1-np.min(Sxx_log1)

    dettimes=analyzer_j.df.peak_time-utcstart_chunk
    samp_down=math.ceil(dt_down/(t[1]-t[0]))
    samp_up=math.ceil(dt_up/(t[1]-t[0]))
    #ones_pad=np.ones((1,pad_length))*np.median(env)
    ones_pad=np.zeros((1,pad_length))

    
    env_pad=np.concatenate((ones_pad[0],env,ones_pad[0]))
    #stacked_env=np.zeros((1,samp_down+samp_up))
    
    maxamp=[]
    medamp = np.median(env)
    snr = []
    ambient_snr = []

    for det in dettimes:
        
        env_chunk=np.zeros((1,samp_down+samp_up))
        #import pdb; pdb.set_trace()
        timeind1=(np.abs(t - det)).argmin()
        timeind=timeind1+pad_length
        startind=timeind-samp_up
        endind=timeind+samp_down

        env_chunk = env_pad[startind:endind] #grab spectrogram subset for multiplication

        #stacked_env=np.add(stacked_env,env_chunk)
        maxamp += [max(env_chunk)]
        med_env_chunk = np.median(env_chunk)
        ambient_snr += [10*np.log10(med_env_chunk/medamp)]
        snr += [10*np.log10(max(env_chunk)/medamp)]
        #import pdb; pdb.set_trace()
        
    
    #stacked_t=np.transpose(range(len(stacked_env[0]))*(t[1]-t[0]))
    return [maxamp,ambient_snr,snr,medamp]

def get_snr(analyzer_j,t,f,Sxx,utcstart_chunk,snr_limits=[14, 16],snr_calllength=4,snr_freqwidth=.6,dur=10,fs=100,window_len=1):

    peak_times=analyzer_j.df['peak_time'].to_list()
    snr=[]
    #import pdb; pdb.set_trace()
    freq_inds=np.where(np.logical_and(f>=min(snr_limits), f<=max(snr_limits)))
    Sxx_sub=Sxx[freq_inds,:][0]
    f=f[freq_inds]

    med_noise = np.median(Sxx_sub)
    utc_t = [utcstart_chunk + j for j in t]   
    snr_t_int=np.int64((snr_calllength/2)/(utc_t[1] - utc_t[0]))
    db_amps = []

    for utc_time in peak_times:
        

        t_peak_ind = utc_t.index(utc_time) 
        Sxx_t_inds1 = list(range(t_peak_ind-snr_t_int,t_peak_ind+snr_t_int))
        #import pdb; pdb.set_trace()
        Sxx_t_inds = [x for x in Sxx_t_inds1 if x < len(t)]
        Sxx_t_sub = Sxx_sub[:,Sxx_t_inds]
        db_max = np.max(Sxx_sub[:,t_peak_ind])
        max_loc = np.where(Sxx_sub[:,t_peak_ind] == db_max)
        freq_max=f[max_loc]
        #f_inds = np.where(np.logical_and(f>=freq_max-snr_freqwidth/2, f<=freq_max+snr_freqwidth/2))
        f_inds = np.where(np.logical_and(f>=snr_limits[0], f<=snr_limits[1]))
        Sxx_tf_sub = Sxx_t_sub[f_inds,:]
        call_noise = np.median(Sxx_tf_sub)
        ##############Make amplitude calculations######
        noise_scale = np.mean(Sxx_tf_sub)/(window_len*(fs**2))
        amp = np.sqrt(noise_scale*(snr_limits[1]-snr_limits[0]))
        db_amps += [10.*np.log10(amp)]
        snr = snr + [call_noise/med_noise]
        #import pdb; pdb.set_trace()

    #Get SNR of 5 seconds of noise preceding call
    start_times=analyzer_j.df['start_time'].to_list()
    noise_t_int=np.int64((dur/2)/(utc_t[1] - utc_t[0]))
    start_snr=[]
    for utc_time in start_times:
        
        t_peak_ind = utc_t.index(utc_time) 
        Sxx_t_inds1 = list(range(t_peak_ind-noise_t_int-2,t_peak_ind-2)) #5 second of noise ending 2 seconds prior to call start time
        #import pdb; pdb.set_trace()
        Sxx_t_inds = [x for x in Sxx_t_inds1 if x >= 0]
        Sxx_t_sub = Sxx_sub[:,Sxx_t_inds]
        ambient_noise = np.median(Sxx_t_sub)
        start_snr = start_snr + [10*np.log10(ambient_noise/med_noise)]

    #Get SNR of 3 seconds after call
    #end_times=analyzer_j.df['end_time'].to_list()
    #noise_t_int=np.int64((dur)/(utc_t[1] - utc_t[0]))
    #end_snr=[]
    #for utc_time in end_times: 
        #t_peak_ind = utc_t.index(utc_time) 
        #Sxx_t_inds1 = list(range(t_peak_ind,t_peak_ind+noise_t_int))
        #Sxx_t_inds = [x for x in Sxx_t_inds1 if x < len(t)]
        #Sxx_t_sub = Sxx_sub[:,Sxx_t_inds]
        #ambient_noise = np.median(Sxx_t_sub)
        #end_snr = end_snr + [10*np.log10(ambient_noise/med_noise)]


    #average SNR from start and end
    #import pdb; pdb.set_trace()
    #ambient_snr=[(start_snr[j]+end_snr[j])/2 for j in range(len(start_snr))]
    ambient_snr=[start_snr[j] for j in range(len(start_snr))]
    
    return snr, ambient_snr, db_amps


def freq_analysis(analyzer_j,t,f,Sxx,utcstart_chunk,freq_window=[14, 16.5]):
    peak_freqs = []
    start_freqs = []
    end_freqs = []
    peak_stds = []
    start_stds = []
    end_stds = []
    #import pdb; pdb.set_trace()
    freq_inds = np.where(np.logical_and(f>=min(freq_window), f<=max(freq_window)))
    Sxx_sub = Sxx[freq_inds,:][0]
    f = f[freq_inds]
    utc_t = [utcstart_chunk + j for j in t]   
    utc_array = np.array(utc_t)

    for k in range(0,len(analyzer_j.df)): #for peak freq
        
        starttime = analyzer_j.df['start_time'][k]-2
        endtime = analyzer_j.df['end_time'][k]+2
        callbool = (utc_array < endtime) & (utc_array > starttime)
        inds = np.array(list(range(len(callbool))))
        callinds = inds[callbool]
        call_times = utc_array[callbool]
        Sxx_total_sub = Sxx_sub[:,callinds]
        farray = np.matlib.repmat(f,len(call_times),1)
        peak_freq = np.sum(np.multiply(farray.T,Sxx_total_sub))/np.sum(Sxx_total_sub)
        peak_freqs = peak_freqs + [peak_freq]
        peak_std = np.sum(np.multiply(np.power(np.subtract(farray.T,np.mean(farray)),2),Sxx_total_sub))/np.sum(Sxx_total_sub)
        peak_stds = peak_stds + [peak_std]
        

    for k in range(0,len(analyzer_j.df)): #for start freq
        
        starttime = analyzer_j.df['start_time'][k]
        endtime = analyzer_j.df['start_time'][k]+1
        callbool = (utc_array < endtime) & (utc_array > starttime)
        inds = np.array(list(range(len(callbool))))
        callinds = inds[callbool]
        call_times = utc_array[callbool]
        Sxx_total_sub = Sxx_sub[:,callinds]
        farray = np.matlib.repmat(f,len(call_times),1)
        start_freq = np.sum(np.multiply(farray.T,Sxx_total_sub))/np.sum(Sxx_total_sub)
        start_freqs = start_freqs + [start_freq]
        start_std = np.sum(np.multiply(np.power(np.subtract(farray.T,np.mean(farray)),2),Sxx_total_sub))/np.sum(Sxx_total_sub)
        start_stds = start_stds + [start_std]

    for k in range(0,len(analyzer_j.df)): #for end freq
        
        starttime = analyzer_j.df['end_time'][k]-1
        endtime = analyzer_j.df['end_time'][k]
        callbool = (utc_array < endtime) & (utc_array > starttime)
        inds = np.array(list(range(len(callbool))))
        callinds = inds[callbool]
        call_times = utc_array[callbool]
        Sxx_total_sub = Sxx_sub[:,callinds]
        farray = np.matlib.repmat(f,len(call_times),1)
        end_freq = np.sum(np.multiply(farray.T,Sxx_total_sub))/np.sum(Sxx_total_sub)
        end_freqs = end_freqs + [end_freq]
        end_std = np.sum(np.multiply(np.power(np.subtract(farray.T,np.mean(farray)),2),Sxx_total_sub))/np.sum(Sxx_total_sub)
        end_stds = end_stds + [end_std]


    return peak_freqs, start_freqs, end_freqs, peak_stds, start_stds, end_stds