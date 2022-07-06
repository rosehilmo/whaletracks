#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:40:53 2021

@author: Xuyang
"""

import pandas as pd
import numpy as np

#create library of songs based on file
AXB_FILE='Fins_chunk_AXBA1_2016_2hr_250.csv'

THRESHOLD=[2000]

freq_lib = {}

A_df0=pd.read_csv(AXB_FILE) #load auto_picked calls
A_df=A_df0[['peak_time', 'peak_epoch', 'peak_signal','peak_frequency']].copy()
print(A_df.info(), A_df.head(5))