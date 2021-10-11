
import sys

sys.path.append('/Users/wader/Desktop/whaletracks/') #allows us to import my written function 

#imports neccessary functions for code
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read, read_inventory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import sys
import os
import pickle
from datetime import datetime
import scipy.signal as sig
import whaletracks.detection.detect_manual as detect
from whaletracks.common.util import datestrToEpoch

#Define constants
HALF_HOUR = 1800  # in seconds
CHUNK_LENGTH=HALF_HOUR/2  #Sets spectrogram time chunk. 15 minutes is ideal for selecting blue whale calls.
PLOTFLAG=True #keep this true since you are manually picking
CHUNK_FILE='Blue_picks.csv' #csv that is extended with each time period
detection_pth="Final_blue_picks.csv" #final .csv name
is_restart='True' #Keep this true to use chunk file
CLIENT_CODE = 'IRIS' #obspy gets data from IRIS
network="7D" #Network name "OO" for OOI, "7D" for Cascadia
station_id_list=['FN14A']#Pick 1 station at a time
location='*'  # '*' for all available locations
channel='HHZ' #Choose channel from specific station,  you'll want 'BHZ,HHZ' for Cascadia
                      #Check http://ds.iris.edu/mda/OO/ for OOI station channels

#Select start and end times for manual picking
STARTTIME = ("2011-12-15T10:35:00.000")
ENDTIME = ("2011-12-15T11:20:00.000")


##########################################################################################
#################End of constants. Change beyond this point at your own risk!###################################
##########################################################################################

utcstart = UTCDateTime(STARTTIME)
utcend = UTCDateTime(ENDTIME)

utcstart_chunk = utcstart
utcend_chunk = utcstart + CHUNK_LENGTH    

client=Client(CLIENT_CODE)

#Defines dataframe structure: don't mess with this
df_columns=['start_time','start_epoch','start_frequency','snr','call_type','ambient_snr','station','channel','peak_frequency','peak_frequency_std']

while utcend > utcstart_chunk:

    #Either load working verification file or make a new one
    if os.path.isfile(CHUNK_FILE) and is_restart:
        Blue_calls = pd.read_csv(CHUNK_FILE)
    else:
        Blue_calls = pd.DataFrame(columns=df_columns)
        

    for station_ids in station_id_list:  #loops through stations requested. Reccomend 1 station at a time however. 
        st_raw_exist = False
        retry=0
        
        #import pdb; pdb.set_trace()

        #Get waveform from IRIS
        while st_raw_exist == False and retry < 5:
                try:
                    st_raw=client.get_waveforms(network=network, station=station_ids, location=location,
                                                channel=channel, starttime=utcstart_chunk,
                                                endtime=utcend_chunk, attach_response=True)
                    st_raw_exist=True
                    retry=5
                except: #If request fails, try again up to 5 times
                    retry=retry+1
                    st_raw_exist=False
                    print("Client failed: Retry " + str(retry) + " of 5 attempts")

        #If data does not exist, move on to next time chunk            
        if st_raw_exist == False:
                print("WARNING: no data available from input station/times")
                utcstart_chunk=utcstart_chunk+CHUNK_LENGTH
                utcend_chunk=utcend_chunk+CHUNK_LENGTH
                
                continue

        #import pdb; pdb.set_trace()
        #Filter waveform, remove response, remove sensitivity
        st_raw.detrend(type="demean")
        st_raw.detrend(type="linear")
        st_raw.remove_response(output='VEL',pre_filt=[.2,.5,18,20])
        st_raw.remove_sensitivity()

        tr=st_raw[0]

        #specifically select waveform            
        tr_filt=tr.copy()

        #Make spectrogram of data
        [f,t,Sxx]=detect.plotwav(tr_filt.stats.sampling_rate, tr_filt.data, window_size=8, overlap=.98, plotflag=False)
        

        #Subsample spectrogram to call range and convert to dB
        freq_inds=np.where(np.logical_and(f>=13, f<=16.5))
        f_sub=f[freq_inds]
        Sxx_sub=Sxx[freq_inds,:][0]
        Sxx_log1=10*np.log10(Sxx_sub)
        Sxx_log=Sxx_log1-np.min(Sxx_log1)

        #Choose color range for spectrogram
        vmin=np.median(Sxx_log)+2*np.std(Sxx_log)
        vmax=np.median(Sxx_log)

        
        if PLOTFLAG==True:
                
            t1=min(t)
            t2=max(t)
        #Make figure and plot axis
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(111)
            
        #plot spectrogram and prepare for user input
            cmap = plt.get_cmap('viridis')
            norm = color.Normalize(vmin=vmin, vmax=vmax)
            im = ax1.pcolormesh(t, f_sub, Sxx_log, cmap=cmap,norm=norm) 
            fig.colorbar(im, ax=ax1,orientation='horizontal')
            ax1.set_xlim([t1, t2]) #look at spectrogram segment between given time boundaries
            ax1.set_ylim([13, 16.5])
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_xlabel('Seconds past ' + UTCDateTime.strftime(utcstart_chunk,'%Y-%m-%dT%H:%M:%S.%fZ'))
            ax1.set_title('Select calls')
            fig.tight_layout()

            #Request user input to select calls
            #Left click on observed calls as prompted 
            #Right click to remove previous pick if you make a mistake
            print('Select all B-calls at starting time and frequency')
            B_calls=plt.ginput(n=-1,timeout=-1)
            print('Select all A-calls at starting time and frequency')
            A_calls=plt.ginput(n=-1,timeout=-1)
        
        #Format B-calls
        b_picks=[]
        b_freq=[]
        for inds in range(0,len(B_calls)):
            ind=inds-1
            b_picks=b_picks+[utcstart_chunk+B_calls[ind][0]]
            b_freq=b_freq+[B_calls[ind][1]]
        b_epochs=detect.datetimeToEpoch(b_picks)
        btype=list(np.repeat('B', len(b_epochs)))

        #Format A-calls
        a_picks=[]
        a_freq=[]
        for inds in range(0,len(A_calls)):
            ind=inds-1
            a_picks=a_picks+[utcstart_chunk+A_calls[ind][0]]
            a_freq=a_freq+[A_calls[ind][1]]
        a_epochs=detect.datetimeToEpoch(a_picks)   
        atype=list(np.repeat('A', len(a_epochs))) 

        #Combine A and B picks
        all_picks=b_picks+a_picks
        all_epochs=b_epochs+a_epochs
        all_freq=b_freq+a_freq
        all_type=btype+atype
        
        #SNR analysis of manual detections
        [snr,ambient_snr] = detect.get_snr(all_picks, t, f_sub, Sxx_sub, utcstart_chunk)
        #Frequency analysis of manual detections
        [peak_freq, freq_std] = detect.freq_analysis(all_picks, all_type, t, f_sub, Sxx_sub, utcstart_chunk)

        #Makes dictionary of detections 
        dct = {k: [] for k in df_columns}
        for index in range(0,len(all_picks)):

            dct['start_time'].append(all_picks[index])
            dct['start_epoch'].append(all_epochs[index])
            dct['start_frequency'].append(all_freq[index])
            dct['call_type'].append(all_type[index])
            dct['station'].append(station_ids)
            dct['channel'].append(channel)
            dct['snr'].append(snr[index])
            dct['ambient_snr'].append(ambient_snr[index])
            dct['peak_frequency'].append(peak_freq[index])
            dct['peak_frequency_std'].append(freq_std[index])
            #dct['snr'].append(None)
            #dct['ambient_snr'].append(None)

        #converts detection dictionary into a dataframe, and appends it to the dataframe from previous chunks
        Blue_calls_sta=pd.DataFrame(dct)
        Blue_calls = Blue_calls.append(Blue_calls_sta)
        #import pdb; pdb.set_trace()
        Blue_calls.to_csv(CHUNK_FILE, index=False) #writes .csv chunk file
        
        plt.close() #closes plots

    utcstart_chunk=utcstart_chunk+CHUNK_LENGTH #moves start chunk to next time period
    utcend_chunk=utcend_chunk+CHUNK_LENGTH  #moves end chunk to next time period

if len(Blue_calls) == 0:
        print('WARNING: detections dataframe empty')
        final_analyzer_df = []

#Writes final .csv file      
else:
    final_analyzer_df = Blue_calls
    final_analyzer_df.to_csv(detection_pth,index=False) 


