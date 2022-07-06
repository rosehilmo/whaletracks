#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:59:44 2021

@author: Xuyang
"""
# Tutorial for plotting topography   https://towardsdatascience.com/plotting-regional-topographic-maps-from-scratch-in-python-8452fd770d9d
# https://github.com/earthinversion/plotting_topographic_maps_in_python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from scipy.interpolate import griddata
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#import the data from local file
df=pd.read_csv('latlon_coor.csv')
df.columns=['longitude','latitude','depth']

Lonmin,Lonmax = min(df.longitude)+2, max(df.longitude)-2
Latmin,Latmax = min(df.latitude), max(df.latitude)
pts=1000

#Some points to be shown on figure
#station AXBA1
AXBlat,AXBlon = 45.820179, -129.736694
#station HYSB1
HYSlat,HYSlon = 44.509781, -125.405296
#Portland for reference
#Ptllat,Ptllon = 45.523064, -122.676483
#import pdb; pdb.set_trace();
extent = [Lonmin, Lonmax, Latmin,Latmax]

#Plotting
fig, ax = plt.subplots(figsize=(10,8), dpi=80, subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent(extent)

ax.plot(AXBlon, AXBlat, 'b*',label='AXBA1',markersize=20)
ax.plot(HYSlon, HYSlat, 'r*',label='HYSB1',markersize=20)
#ax.plot(Ptllon, Ptllat, 'k*',label='Portland',markersize=20)
ax.text(AXBlon-0.7, AXBlat+0.2, 'AXBA1',fontsize = 25)
ax.text(HYSlon-0.7, HYSlat+0.2, 'HYSB1',fontsize = 25)
#ax.text(Ptllon-0.7, Ptllat+0.2, 'Portland',fontsize = 25)

#Adding the topography layer
[x,y] = np.meshgrid(np.linspace(Lonmin, Lonmax, pts),
                    np.linspace(Latmin, Latmax, pts))
z = griddata((df.longitude, df.latitude), df.depth, (x, y), method='linear');
x = np.matrix.flatten(x); #Gridded longitude
y = np.matrix.flatten(y); #Gridded latitude
z = np.matrix.flatten(z); #Gridded elevation
#import pdb; pdb.set_trace();
im = ax.scatter(x,y,1,z,cmap='terrain')

#add features to map
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.STATES)
ax.coastlines()

#adjust the axis labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels=False
gl.right_labels=False
gl.xlocater = mticker.FixedLocator([-133, -130, -127, -124, -121])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':15}
gl.ylabel_style = {'size':15}

#add a colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, orientation='horizontal')
cbar.set_label(label='Elevation above sea level [m]')
    
#ax.set_title('Area of Interest', fontsize=25)
#import pdb; pdb.set_trace();
plt.show()
'''
plt.scatter(x,y,1,z,cmap='terrain')
plt.plot(AXBlon, AXBlat, 'r^',label='AXBA1')
plt.plot(HYSlon, HYSlat, 'b^',label='HYSB1')
plt.colorbar(label='Elevation above sea level [m]')
plt.xlim(min(df.longitude),max(df.longitude))
plt.ylim(min(df.latitude),max(df.latitude))
plt.xlabel('Longitude [°]')
plt.ylabel('Latitude [°]')
#plt.legend()
plt.show()

'''