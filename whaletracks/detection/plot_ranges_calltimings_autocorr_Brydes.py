

import sys

sys.path.append('/Users/rhilmo/Documents/GitHub/whaletracks/') 

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from whaletracks.common.util import datetimeToEpoch
from obspy import UTCDateTime
import numpy as np
import math
import basic_ranging_model as ranging
from datetime import timedelta
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

VER_FLAG=False
BELLHOP=True
REFLECTIVITY=False

if BELLHOP == False:

    #depth=4284; ss=1494; sed_speed=1730; t=1000; #depth in meters in AK LD41

    #depth=4550; ss=1496; sed_speed=1730; t=1000; #depth in meters in AK WD59

    #depth=4626; ss=1497; sed_speed=1700; t=250; #depth in meters in AK WD62

    #epth=4410; ss=1503; sed_speed=0; t=0; #average depth and sound speed m/s Hawaii

    #depth=5154; ss=1515; sed_speed=2000; t=1600; #average depth and sound speed m/s ENAM X06

    depth=5271; ss=1512; sed_speed=1550; t=419; #average depth and sound speed m/s ENAM X08

    #depth=5949; ss=1512; sed_speed=1500; t=250 #t=36; #Marainas

    #depth=5979+120; ss=1512; sed_speed=0; t=0 #t=36; #Marainas

    #depth=1949; ss=1450; sed_speed=0; t=0; #Momar

    #depth=817; ss=1490; sed_speed=0; t=0; #Hydrate Ridge

    #depth=4860; ss=1515; sed_speed=0; t=0 #t=36; #Portugal

    #depth=5426; ss=1515; sed_speed=2000; t=176; #Puerto Rico

    #starttime=UTCDateTime("2019-02-09T06:36:00.000")

    [distance,t0,t1,t2,t3,t0_1,t1_1,t1_2]=ranging.basic_ranging(depth,ss,sed_speed,t,plotflag=False)
    mp_1_timing=np.subtract(t1_1,t0)
    #p_1_1_timing=np.subtract(t1_1,t0)
    mp_2_timing=np.subtract(t2,t1)

if BELLHOP == True:
    #df_bellhop=pd.read_csv('Marianas_bellhop_arrivals_40km.csv')
    df_bellhop=pd.read_csv('Marianas_bellhop_arrivals_40km.csv')
    distance=df_bellhop['interp_r']
    t0=df_bellhop['interp_d']
    t1=df_bellhop['interp_mp1']
    t2=df_bellhop['interp_mp2']
    t3=df_bellhop['interp_mp3']
    mp_1_timing=np.subtract(t1,t0)
    mp_2_timing=np.subtract(t2,t1)
    mp_3_timing=np.subtract(t3,t2)
    if REFLECTIVITY == True:
        mp_1_sub=df_bellhop['interp_sub']
        

#mp_1_timing[0:1000]=mp_1_1_timing[0:1000]

#peaks=pickle.load(open("peaks_B19_mp_edit_stack.p","rb"))
#minutes1=np.linspace(1,len(peaks[0]),len(peaks[0]))
#plt.pcolormesh(minutes1,peaks[2],np.log10(np.transpose(peaks[1])),cmap='gist_ncar')
#plt.pcolormesh(minutes1,peaks[2],np.log10(np.transpose(np.add(peaks[1],np.abs(np.min(peaks[1]))))),cmap='gist_ncar')
#minep=datetimeToEpoch([starttime]) 
#plt.xticks(ticks=minutes1[55::60*3], labels=peaks[0][55::60*3])
#import pdb; pdb.set_trace()

t1=0
#df1=pd.read_csv('AKtest_stack_win0_2_auto.csv')  
#df1=pd.read_csv('AKtest_stack_win0_5_10min.csv')  
df_calls=pd.read_csv('B19_mp_Brydes_20min.csv')
df_calls['peak_time']=pd.to_datetime(df_calls['peak_time'])
calltimes=df_calls['peak_time'].unique()

plt.rc('axes', axisbelow=True)
fig,ax = plt.subplots(figsize=(15,6))
date_form = DateFormatter("%H%M")
ax.xaxis.set_major_formatter(date_form)

autotimes_save=[]
autopeaks_mp1_save=[]
autopeaks_mp2_save=[]
autopeaks_mp3_save=[]
auto_max=[]
auto_snr=[]
auto_amp=[]
droprows=[]
auto_count=[]
center_calls = []
low_freq_snr = []

"""
df1=pd.read_csv('stack_B19_mp.csv') 
#minep=min(df1['peak_epoch'])
maxminute=(max(df1['epoch'])-min(df1['epoch']))/60
datetimes=df1['date'].tolist()
dates=[pd.to_datetime(d) for d in datetimes]
calltimes=df_calls['peak_time'].unique()



#plt.grid(axis='both')
for row in range(len(df1)):
    #import pdb; pdb.set_trace()
    df=df1.iloc[row]
    t2=df['arrival_2']-df['arrival_1']
    date=dates[row]
    start_center_min=date-timedelta(seconds=30)
    end_center_min=date+timedelta(seconds=30)
    unique_calls=calltimes[(calltimes>start_center_min) & (calltimes<end_center_min)]
    #unique_calls=sub_df['peak_time'].unique()
    #import pdb; pdb.set_trace()
    if len(unique_calls) > 0:

        ep=df['epoch'] 
        #minutes=(ep-minep)/60 


        mpdiff_1=df['arrival_2']-df['arrival_1'] 
        mpdiff_2=df['arrival_3']-df['arrival_1'] 
        mpdiff_3=df['arrival_4']-df['arrival_1'] 
        mpdiff_4=df['arrival_5']-df['arrival_1'] 

        timings=[mpdiff_1,mpdiff_2,mpdiff_3,mpdiff_4]
        amps=[df['amp_2'],df['amp_3'],df['amp_4'],df['amp_5']]
        k_max=max(amps)
        max_ind=amps.index(k_max)
        mp_timing=timings[max_ind]
        near_timings=[abs(mp-mp_timing) for mp in mp_1_timing]
        near_timings_2=[abs(mp-mp_timing) for mp in mp_2_timing]
        best_match=min(near_timings)
        best_match_2=min(near_timings_2)
        match_ind=near_timings.index(best_match)
        match_ind_2=near_timings_2.index(best_match_2)
        best_distance=distance[match_ind]
        best_distance_2=distance[match_ind_2]

        if best_distance < 110 and REFLECTIVITY == True:
            near_timings_sub=[abs(mp-mp_timing) for mp in mp_1_sub]
            best_match_sub=min(near_timings_sub)
            match_ind_sub=near_timings_sub.index(best_match_sub)
            best_distance_sub=distance[match_ind_sub]
            plt.scatter(date,best_distance_sub/1000,[70],facecolors='none', edgecolors='magenta',zorder=1) 
            best_distance = best_distance_sub
            #import pdb; pdb.set_trace()
            #autopeaks_mp1_save += [best_distance_sub/1000]
        #else:
            #autopeaks_mp1_save += [best_distance/1000]

        
        #plt.scatter(minutes,best_distance_2,c='cyan',zorder=1) 
        plt.scatter(date,best_distance/1000,[100],facecolors='none', edgecolors='cornflowerblue', zorder=1)  
        
        
        
        

    t1=t2

   """
#import pdb; pdb.set_trace()

t1=0
df1=pd.read_csv('auto_B19_mp_Brydes_20min.csv')  
#df1=pd.read_csv('AKtest_stack_win0_2.csv')  
#minep=min(df1['peak_epoch'])
maxminute=(max(df1['epoch'])-min(df1['epoch']))/60
datetimes=df1['date'].tolist()
dates=[pd.to_datetime(d) for d in datetimes]


for row in range(len(df1)):
    #import pdb; pdb.set_trace()
    df=df1.iloc[row]
    t2=df['arrival_2']-df['arrival_1']
    date=dates[row]
    start_center_min=date-timedelta(seconds=30)
    end_center_min=date+timedelta(seconds=30)
    #sub_df=df_calls.loc[(df_calls['peak_time']>start_center_min) & (df_calls['peak_time']<end_center_min)]
    #calltimes=df_calls['peak_time'].unique()
    unique_calls=calltimes[(calltimes>start_center_min) & (calltimes<end_center_min)]
    #unique_calls=sub_df['peak_time'].unique()
    #import pdb; pdb.set_trace()
    if len(unique_calls) > 0:

        ep=df['epoch'] 
        #minutes=(ep-minep)/60 


        mpdiff_1=df['arrival_2']-df['arrival_1'] 
        mpdiff_2=df['arrival_3']-df['arrival_1'] 
        mpdiff_3=df['arrival_4']-df['arrival_1'] 
        mpdiff_4=df['arrival_5']-df['arrival_1'] 

        
        timings=[mpdiff_1,mpdiff_2,mpdiff_3,mpdiff_4]
        amps=[df['amp_2'],df['amp_3'],df['amp_4'],df['amp_5']]
        k_max=max(amps)
        max_ind=amps.index(k_max)
        mp_timing=timings[max_ind]
        near_timings=[abs(mp-mp_timing) for mp in mp_1_timing]
        near_timings_2=[abs(mp-mp_timing) for mp in mp_2_timing]
        near_timings_3=[abs(mp-mp_timing) for mp in mp_3_timing]
        
        best_match=min(near_timings)
        best_match_2=min(near_timings_2)
        best_match_3=min(near_timings_3)
        
        match_ind=near_timings.index(best_match)
        match_ind_2=near_timings_2.index(best_match_2)
        match_ind_3=near_timings_3.index(best_match_3)
        
        best_distance=distance[match_ind]
        best_distance_2=distance[match_ind_2]
        best_distance_3=distance[match_ind_3]
        
        
        #plt.scatter(minutes,best_distance_2,c='grey',zorder=1) 
        #plt.scatter(date,best_distance/1000,c='navy',zorder=1) 

        if best_distance < 110 and REFLECTIVITY == True:
            near_timings_sub=[abs(mp-mp_timing) for mp in mp_1_sub]
            best_match_sub=min(near_timings_sub)
            match_ind_sub=near_timings_sub.index(best_match_sub)
            best_distance_sub=distance[match_ind_sub]
            plt.scatter(date,best_distance_sub/1000,c='pink',zorder=1) 
            best_distance = best_distance_sub
            #import pdb; pdb.set_trace()
            autopeaks_mp1_save += [best_distance_sub/1000]
        else:
            autopeaks_mp1_save += [best_distance/1000]
        
        
        autotimes_save += [date]
        auto_max += [(df['peaks'])]
        auto_snr += [(df['snr'])]
        auto_amp += [(df['db_amps'])]
        auto_count += [(df['sum_calls'])]
        center_calls += [(df['n_calls'])]
        low_freq_snr += [(df['low_snr'])]
        autopeaks_mp2_save += [best_distance_2/1000]
        autopeaks_mp3_save += [best_distance_3/1000]
        
    if t1 == t2: 
        droprows += [row]
    t1=t2

df1.drop(index=droprows,inplace=True)  
dictrange = {'time': autotimes_save, 'range_D_MP1': autopeaks_mp1_save, 'range_MP1_MP2': autopeaks_mp2_save, 'range_MP2_MP3': autopeaks_mp3_save, 'auto_max': auto_max, 'auto_snr': auto_snr, 'auto_amp': auto_amp, 'auto_count': auto_count, 'n_calls': center_calls}
saveranges=pd.DataFrame(dictrange)
#saveranges.to_csv('Momar_ranges_11_30_constant_2150m.csv', index=False)
plt.rcParams['font.size'] = '16'
parameters = {'axes.labelsize': 16, 'axes.titlesize': 16}
plt.rcParams.update(parameters)

"""
#import pdb; pdb.set_trace()    
plt.ylim(0,40)
#plt.title('AK range estimations using direct and MP1. Black: Autocorrelation, Blue: Stacking',fontsize=20) 
plt.xlabel('Time',fontsize=16)
plt.ylabel('Distance (km)',fontsize=16)
plt.grid(axis='both')
plt.show() 
plt.close()
"""

#import pdb; pdb.set_trace()
enough_samps = np.where(np.array(auto_max) > 3)
no_low_noise = np.where(np.array(low_freq_snr) < 10)
suminds = np.intersect1d(enough_samps,no_low_noise)


#suminds = np.where(np.array(auto_max) > 3)

timesarray = np.array(autotimes_save)
peaksarray1 = np.array(autopeaks_mp1_save)
peaksarray2 = np.array(autopeaks_mp2_save)
peaksarray3 = np.array(autopeaks_mp3_save)
#colarray = np.array(np.log(auto_max))
colarray = np.array(auto_amp)

fig,ax = plt.subplots(figsize=(13,6))
plt.scatter(timesarray[suminds],peaksarray1[suminds],c=colarray[suminds],cmap="gist_ncar") 
plt.scatter(timesarray[suminds],peaksarray2[suminds],c=colarray[suminds],marker='x',cmap="gist_ncar") 
plt.scatter(timesarray[suminds],peaksarray3[suminds],c=colarray[suminds],marker='+',cmap="gist_ncar") 
#plt.scatter(autotimes_save[j],autopeaks_mp2_save[j],c="magenta",zorder=1)  
plt.grid(axis='both')
plt.title('Attempted ranges for mp1 and mp2 timings, Colors are log(amplitudes) of strongest arrival')
plt.xlabel('Day and hour')
plt.ylabel('Distance (meters)')
plt.ylim(0,40)
plt.xlabel('Time',fontsize=16)
plt.ylabel('Distance (km)',fontsize=16)
plt.rcParams['font.size'] = '16'
parameters = {'axes.labelsize': 16, 'axes.titlesize': 16}
plt.rcParams.update(parameters)
plt.colorbar()
date_form = DateFormatter("%m/%d %H%M")
ax.xaxis.set_major_formatter(date_form)
plt.show()


saveranges.to_csv('Marianas_auto_B19_v2.csv', index=False)

