#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:30:41 2021

@author: erlend
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:16:54 2021

@author: erlend
"""

import datetime
import numpy as np
import pandas as pd
from radar import radar_reflectivity_thredds

    
lat_indice = [1531, 1537]
lon_indice = [577, 582]

start = '2018-07-01' 
stop = '2018-09-11'
#CML_stop = '2018-09-11'
time_start = datetime.datetime.strptime(start, '%Y-%m-%d')
time_stop = datetime.datetime.strptime(stop, '%Y-%m-%d')
daterange = pd.date_range(time_start, time_stop).tolist()
# CML start = '2018-08-01' 
# CML end = '2018-09-11'

reflectivity = []
times = []
for date in daterange:
    ds = radar_reflectivity_thredds(date, lat_indice, lon_indice)
    
    #downloads a radar image for every 7,5 minute
    data = ds.variables['equivalent_reflectivity_factor'][:].data
    data[data == 9.96921e+36] = 0 # zero rainfall
    
    for hour in data:
        reflectivity.append( hour.ravel()  ) # grid as vector
    for hour in ds.variables['time'][:].data: 
    #time as number of seconds since epoch
        times.append(hour) # grid as vector

reflectivity = np.array(reflectivity)
times = np.array(times).reshape(-1, 1)
#make pandas dataframe and set index as col number 0
df = pd.DataFrame(np.hstack([times, reflectivity])).set_index(0)
df.to_pickle('radar_larger.pkl')
        