#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:01:51 2021

@author: erlend
"""
import requests
import numpy as np
import pandas as pd
import datetime


def get_rain_from_frost_hourly(ids, start, end, elements="sum(precipitation_amount PT1H)", clientID="6e7d29f6-8f77-4317-b9a6-846a7052852c"):
    url = "https://frost.met.no/observations/v0.jsonld"
    reftime = f"{start}/{end}"
    headers = {"Accept": "application/json"}
    parameters = {
        "sources": ids,
        "referencetime": reftime,
        "elements": elements,
        "timeoffsets": "PT0H",
        "fields": "sourceId, referenceTime, value, elementId",
    }
    r = requests.get(url=url, params=parameters, headers=headers, auth=(clientID, ""))
    return r.json()



if __name__ == "__main__":
    # CML start = '2018-08-01' 
    # CML end = '2018-09-11'
    start = '2018-07-01' 
    end = '2018-09-11'
    met_stations = {
        "Rustadskogen": {
            "shortname": "Rustadskogen",
            "id": "SN17870",
            "name": "ÅS - RUSTADSKOGEN",
            "maxResolution": "PT1H", #"PT1M",
            "lat": "59.6703",
            "lon": "10.8107" 
    }}
    station = get_rain_from_frost_hourly(met_stations['Rustadskogen']['id'], start, end)['data']
    t = []
    p = []
    
    for obs in station:
        t.append(datetime.datetime.strptime(obs['referenceTime'][0:13], '%Y-%m-%dT%H'))
        p.append(obs['observations'][0]['value'])
    
    t = np.array(t).reshape(-1, 1)
    p = np.array(p).reshape(-1, 1)
    
    df = pd.DataFrame(np.hstack([t, p])).set_index(0)
    df.to_pickle('rustad_weather_station.pkl') 
    #df = pd.DataFrame.from_dict(data)    
    #df_sort = df.pivot_table(2, [0, 1])




# =============================================================================
#     id_stasjon = met_stations[i]["id"]
#     maxResolution = met_stations[i]["maxResolution"]
#     for i in get_rain_from_frost_hourly(id_stasjon, start, end)['data']:
#         data.append( [id_stasjon, i['referenceTime'], i['observations'][0]['value']] )
# =============================================================================



