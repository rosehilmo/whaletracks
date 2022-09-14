#/opt/anaconda3/envs/whaletracks python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:44:06 2020

@author: wader
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
import matplotlib.colors as color
import sys
import os
sys.path.append('/Users/wader/Desktop/whaletracks/whaletracks/') 
from common.util import datestrToEpoch

#Define constants
CLIENT_CODE = 'IRIS'
HALF_HOUR = 1800  # in seconds
CHUNK_LENGTH=HALF_HOUR/6   #seconds in spectrogram
PLOTFLAG=True #True makes plots
PLT_SCORE=4 #figure number
CHUNK_FILE='Fins_verified_Alaska.csv' #File name where verified calls will be saved
DETECTION_FILE='AlaskaFins_highpulse_temp.csv' #File name produced by main_detection
is_restart='True' #each iteration reads CHUNK_FILE and adds new verified calls
station_id_list=['WD50'] #List each station you want to verify

STARTTIME = ("2018-10-06T08:30:00.000") #start time

STARTEPOCH = datestrToEpoch([STARTTIME],dateformat='%Y-%m-%dT%H:%M:%S.%f')
ENDEPOCH = [STARTEPOCH[0] + CHUNK_LENGTH]

utcstart_chunk = UTCDateTime(STARTTIME)
utcend_chunk = UTCDateTime(STARTTIME) + CHUNK_LENGTH

client=Client('IRIS')

#Load call dataframes
b_df=pd.read_csv(DETECTION_FILE)


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
    b_df_sub=b_df.loc[(b_df['station_code'] == station_ids) & 
                        (b_df['peak_epoch'] > STARTEPOCH[0]) &
                        (b_df['peak_epoch'] < ENDEPOCH[0])]


    #Get waveform from IRIS
    while st_raw_exist == False and retry < 5:
            try:
                st_raw=client.get_waveforms(network="XO", station=station_ids, location='*',
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
    st_raw.remove_response(output='VEL',pre_filt=[.2,.5,45,48])
    st_raw.remove_sensitivity()

    tr=st_raw[0]

    #specifically select waveform            
    tr_filt=tr.copy()

    #Make spectrogram of data
    [f,t,Sxx]=plotwav(tr_filt.stats.sampling_rate, tr_filt.data, filt_freqlim=[12, 40],window_size=1, overlap=.95, plotflag=False)
    t=t+STARTEPOCH 

    #Subsample spectrogram to call range and convert to dB
    freq_inds=np.where(np.logical_and(f>=12, f<=40))
    f_sub=f[freq_inds]
    Sxx_sub=Sxx[freq_inds,:][0]
    Sxx_log1=10*np.log10(Sxx_sub)
    Sxx_log=Sxx_log1-np.min(Sxx_log1)

    #Choose color range for spectrogram
    vmin=np.median(Sxx_log)+1.5*np.std(Sxx_log)
    vmax=np.median(Sxx_log)

    #Prepare call data for plotting

    b_start=b_df_sub['start_epoch'].values.tolist()
   
    b_end=b_df_sub['end_epoch'].values.tolist()
 
    b_score=b_df_sub['peak_signal'].values.tolist()
    #bfilt_score=bfilt_df_sub['peak_signal'].values.tolist()
   
    b_peak=b_df_sub['peak_epoch'].values.tolist()
    
    


    if PLOTFLAG==True:
            
        t1=min(t)
        t2=max(t)
    #plot a and b calls on upper axis
        #plt.figure(PLT_SCORE, figsize=(9, 3))
        fig, (ax0, ax1) = plt.subplots(nrows=2,sharex=True)

        #plot calls
        for b_ind in range(0,len(b_start)):
            ax0.plot([b_start[b_ind],b_end[b_ind]],[b_score[b_ind],b_score[b_ind]],'y') 
            ax0.plot(b_peak,b_score,'yx')


        ax0.set_xlim([t1, t2]) #look at only positive values
        ax0.set_ylim([0, np.max(b_score+[1])+.2])
        ax0.set_xlabel('Seconds')
        ax0.set_ylabel('Detection score')
        ax0.set_title('Detection score, peak time, and width')

    #plot spectrogram on lower axis
        cmap = plt.get_cmap('viridis')
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        #plt.subplot(212)
        im = ax1.pcolormesh(t, f_sub, Sxx_log, cmap=cmap,norm=norm) 
        fig.colorbar(im, ax=ax1,orientation='horizontal')
        ax1.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
        ax1.set_ylim([12, 40])
        ax1.set_ylabel('Frequency [Hz]')
        #ax1.set_xticks([])
        #ax1.set_xlabel('Time [seconds]')
        fig.tight_layout()

        #Request user input to select calls
        #Left click on all x's on top plot that correctly identify B calls
        #Left click in the time vicinity of any missed B calls on top plot
        print('Select detected calls that are true')
        true_calls=plt.ginput(n=-1,timeout=-1)
        print('Select missed calls that are true')
        missed_calls=plt.ginput(n=-1,timeout=-1)

    quality_list=[]

    for inds in range(0,len(true_calls)):
        ind=inds-1
        diff_list=abs(np.subtract(b_peak,true_calls[ind][0]))
        diff=min(diff_list)
        if diff < 5:
            k=np.argmin(diff_list)
            chunk_verified = chunk_verified.append(b_df_sub.iloc[k])
            quality_list=quality_list+[1]

    df_false=pd.concat([b_df_sub,chunk_verified]).drop_duplicates(keep=False)
    quality_list=quality_list+[0]*len(df_false)

    chunk_verified=chunk_verified.append(df_false)
    chunk_verified['quality']=quality_list

    #import pdb; pdb.set_trace()
    for inds2 in range(0,len(missed_calls)):
        ind2 = inds2-1
        new_row=[None]*len(chunk_verified.columns)
        new_row[-1] = 1
        new_row[-3]=station_ids
        new_row[5]=missed_calls[ind2][0]
        df_len=len(chunk_verified)
        chunk_verified.loc[df_len] = new_row

   

    verified_calls=verified_calls.append(chunk_verified, ignore_index=True)
    plt.close()
    
verified_calls.to_csv(CHUNK_FILE,index=False)






