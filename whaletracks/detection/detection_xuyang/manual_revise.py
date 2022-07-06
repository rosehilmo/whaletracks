#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 06:02:22 2021

@author: Xuyang
"""

import sys

sys.path.append('/Users/xuyang/documents/GitHub/whaletracks/whaletracks') 

from obspy import UTCDateTime
import pandas as pd


CHUNK_FILE='Fins_chunk_manual_HYSB1_0.5_dist2.csv'

m_df = pd.read_csv(CHUNK_FILE)
m_peak = m_df['time'].values.tolist()
m_time = []
for starttime in m_peak:
    utcstart = UTCDateTime(starttime)
    m_time.append(utcstart)
m_df['datetime'] = m_time

m_df.to_csv('Fins_manual_HYSB1_0.5_dist2_revise_2.csv', index=False)