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
import netCDF4
import numpy as np
import pandas as pd

# =============================================================================
# def get_xy_indices(min_lat, max_lat, min_lon, max_lon):
#     link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2018/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20180801.nc?Xc[0:1:2066],lon[0:2242][0:2066],lat[0:2242][0:2066]'
#     ds=netCDF4.Dataset(link)
# 
#     lat_bnds, lon_bnds = [min_lat, max_lat], [min_lon, max_lon]
#         
#     lats = ds.variables['lat'][:] 
#     lons = ds.variables['lon'][:]
#     
#     lat_bool = (lats > lat_bnds[0]) & (lats < lat_bnds[1])
#     lon_bool = (lons > lon_bnds[0]) & (lons < lon_bnds[1])
#         
#     lat_indice = [np.where(lat_bool != False)[0][0], np.where(lat_bool != False)[0][-1]]
#     lon_indice = [np.where(lon_bool != False)[1][0], np.where(lon_bool != False)[1][-1]]
#     
#     return lat_indice, lon_indice
# 
# =============================================================================
def radar_reflectivity_thredds(date, lat_indice, lon_indice):
    """
    By requesting lat and lon indice bounds we can download a small 5x6 pixel
    window for our studt area. This reduces download time drastically. 

    """
    # lat and lon indices
    y = '[' + str(lat_indice[0]) + ':' + str(1) + ':' + str(lat_indice[1]) + ']' 
    x = '[' + str(lon_indice[0]) + ':' + str(1) + ':' + str(lon_indice[1]) + ']' 
    
    
    # 
    day = date.strftime('%d')
    month = date.strftime('%m')
    year = date.strftime('%Y')
    
    # dumb fixes in link as some days have missing values
    if year == '2018' and month == '08' and day == '07':
        time = '183'
        
    elif year == '2018' and month == '07' and (
            day == '03' or day == '04' or day == '13' or day == '14' or day == '18' or day == '21'):
        time = '190'    
    
    elif year == '2018' and month == '07' and (day == '15'):
        time = '188' 
        
    else: 
        time = '191'
        
    # Without lat lon cordinates
    link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/'+ year + '/' + month +'/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.'+year+month+day+'.nc?time[0:1:'+time+'],equivalent_reflectivity_factor[0:1:' + time + ']' + y + x 
    
    # With lat lon coordinates. For verification
    #link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2018/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20180801.nc?lon'+y + x+',lat'+ y + x +',equivalent_reflectivity_factor[0:2:191]' + x + y
    ds=netCDF4.Dataset(link)

    return ds

    
if __name__ == "__main__":
    #min_lat = 59.6592
    #max_lat = 59.6916
    #min_lon = 10.7723
    #max_lon = 10.8258
    
    #link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2018/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20180801.nc?Xc[0:1:2066],lon[0:2242][0:2066],lat[0:2242][0:2066]'
    #ds=netCDF4.Dataset(link)

    #lat_bnds, lon_bnds = [min_lat, max_lat], [min_lon, max_lon]
        
    #lats = ds.variables['lat'][:] 
    #lons = ds.variables['lon'][:]
    
    #lat_bool = (lats > lat_bnds[0]) & (lats < lat_bnds[1])
    #lon_bool = (lons > lon_bnds[0]) & (lons < lon_bnds[1])
        
    #lat_indice = [np.where(lat_bool != False)[0][0], np.where(lat_bool != False)[0][-1]]
    #lon_indice = [np.where(lon_bool != False)[1][0], np.where(lon_bool != False)[1][-1]]
    
    # Indices was found by trial and error
    # By running radar_reflectivity_thredds and select the link that also 
    # returns coordinates you can verify that the lat and lon indices match the
    # decired coordinates above
    #original grid:
    #lat_indice = [1532, 1536]
    #lon_indice = [578, 581]
    
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
        