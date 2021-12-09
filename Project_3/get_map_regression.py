#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:05:22 2021

@author: erlend
"""

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
from radar import radar_reflectivity_thredds


    
lat_indice = [1445, 1697] #found by trial and error
lon_indice = [451, 665]
date = '2018-09-08'
date = datetime.datetime.strptime(date, '%Y-%m-%d')
   
ds = radar_reflectivity_thredds(date, lat_indice, lon_indice)
data = ds.variables['equivalent_reflectivity_factor'][:].data
data[data == 9.96921e+36] = 0 # zero rainfall

data_ = data[44, :, :] # interesting cloud here
    
np.save('radar_regression.npy', data_)
    