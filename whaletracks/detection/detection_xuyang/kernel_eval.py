#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:00:43 2021

@author: Xuyang
"""

import matplotlib.pyplot as plt
import numpy as np

t0 = [1135,507,1151,1144,1148,1544,390]
f0 = [1069,137,283,186,53,24,16]
m0 = [42,37,119,141,180,373,204]

false = []
missed = []

for i in range(0,len(t0)):
    fal_r = f0[i] / (f0[i] + t0[i]) * 100
    mis_r = m0[i] / (m0[i] + t0[i]) * 100
    false.append(fal_r)
    missed.append(mis_r)

print(false, missed)

plt.plot(missed, false, 'ro-')
plt.xlim(0,50)
plt.ylim(0,50)
plt.xlabel('% Missed')
plt.ylabel('% False')
plt.show()
