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
import matplotlib.pyplot as plt
import pandas as pd

def get_xy_indices(min_lat, max_lat, min_lon, max_lon):
    link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2018/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20180801.nc?Xc[0:1:2066],lon[0:2242][0:2066],lat[0:2242][0:2066]'
    ds=netCDF4.Dataset(link)

    lat_bnds, lon_bnds = [min_lat, max_lat], [min_lon, max_lon]
        
    lats = ds.variables['lat'][:] 
    lons = ds.variables['lon'][:]
    
    lat_bool = (lats > lat_bnds[0]) & (lats < lat_bnds[1])
    lon_bool = (lons > lon_bnds[0]) & (lons < lon_bnds[1])
        
    lat_indice = [np.where(lat_bool != False)[0][0], np.where(lat_bool != False)[0][-1]]
    lon_indice = [np.where(lon_bool != False)[1][0], np.where(lon_bool != False)[1][-1]]
    
    return lat_indice, lon_indice

def radar_reflectivity_thredds(date, lat_indice, lon_indice):
    """
    By inserting indices we can download a smaller chunck, decreasing download
    time drastically. 

    """
    y = '[' + str(lat_indice[0]) + ':' + str(1) + ':' + str(lat_indice[1]) + ']' 
    x = '[' + str(lon_indice[0]) + ':' + str(1) + ':' + str(lon_indice[1]) + ']' 
    
    # Without cordinates
    link = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2018/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20180801.nc?equivalent_reflectivity_factor[0:2:191]' + x + y # [0:1:2][0:1:0]'
    # With coordinates
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
    lat_indice = [1532, 1536]
    lon_indice = [578, 581]
    
    CML_start = '2018-08-01' 
    CML_stop = '2018-09-11'
    time_start = datetime.strptime(CML_start, '%Y-%m-%d')
    time_stop = datetime.strptime(CML_stop, '%Y-%m-%d')
    daterange = pd.date_range(time_start, time_stop).tolist()
    # CML start = '2018-08-01' 
    # CML end = '2018-09-11'
    for date in daterange:
        ds = radar_reflectivity_thredds(date, lat_indice, lon_indice)
        
    daterange[2].strftime('%d')daterange[2].strftime('%d')daterange[2].strftime('%d')
    ds = radar_reflectivity_thredds(0, lat_indice, lon_indice)
    

"""
    lats = ds.variables['lat'][:] 
    lons = ds.variables['lon'][:]
    

    lat_bool = (lats > lat_bnds[0]) & (lats < lat_bnds[1])
    lon_bool = (lons > lon_bnds[0]) & (lons < lon_bnds[1])
        
    lon_lat_bool = lat_bool & lon_bool
    
    
    
    
    radar = ds.variables['equivalent_reflectivity_factor'][:]
    radar = radar.data[0, :, :]
    radar[~lon_lat_bool] = np.nan
    
    # remove columns and rows with all nan
    radar = radar[~np.isnan(radar).all(axis=1), :]
    radar = radar[:, ~np.isnan(radar).all(axis=0)]
"""

        