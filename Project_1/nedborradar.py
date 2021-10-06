#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:16:54 2021

@author: erlend
"""

import requests
from io import BytesIO
from skimage import color as sk_color #scikit-learn image reader
from skimage import io as sk_io
from skimage import util as sk_util

class RadarImage(object):
    def __init__(self, region, get_old=False):
        if region == 'Eastern Norway':
            query_nedbor = {
                "BBOX": "57.60629522666670255,6.677604090932648617,61.41849905267286402,14.18101722744072823",
                "CRS": "EPSG:4326",
                "DPI": "96",
                "FORMAT": "image/png",
                "FORMAT_OPTIONS": "dpi:96",
                "HEIGHT": "1751", #1863
                "LAYERS": "radar_precipitation_intensity",
                "MAP_RESOLUTION": "96",
                "REQUEST": "GetMap",
                "SERVICE": "WMS",
                "STYLES": "",
                "TRANSPARENT": "TRUE",
                "VERSION": "1.3.0",
                "WIDTH": "3447",
                "language": "eng"
            }
            query_norge = {
                "BBOX": "57.60629522666670255,6.677604090932648617,61.41849905267286402,14.18101722744072823",
                "CRS": "EPSG:4326",
                "DPI": "96",
                "FORMAT": "image/png",
                "FORMAT_OPTIONS": "dpi:96",
                "HEIGHT": "1751",
                "LAYERS": "kart",
                "MAP_RESOLUTION": "96",
                "REQUEST": "GetMap",
                "SERVICE": "WMS",
                "STYLES": "",
                "TRANSPARENT": "TRUE",
                "VERSION": "1.3.0",
                "WIDTH": "3447",
                "language": "eng"
            }
        self.response_nedbor = requests.get(
            'https://metmaps.met.no/metmaps/default.map?VERSION=1.3.0', 
            query_nedbor)
        
        self.response_norge = requests.get(
            'https://metmaps.met.no/metmaps/default.map?VERSION=1.3.0',
            query_norge)
        
    def get_grayscale(self):
        nedbor_color = sk_color.rgba2rgb(sk_io.imread(
            BytesIO(self.response_nedbor.content)))
        
        no_red = nedbor_color[:, :, 0] < 0.8  # mask urelevant colors
        
        grayscale_nedbor = sk_color.rgb2gray(nedbor_color) #*self.no_red
        
        grayscale_nedbor_invert = sk_util.invert(grayscale_nedbor)
        
        return grayscale_nedbor_invert*no_red 
        
    def get_map(self):
        norge_color = sk_color.rgba2rgb(sk_io.imread(
            BytesIO(self.response_norge.content)))
        nedbor_color = sk_color.rgba2rgb(sk_io.imread(
            BytesIO(self.response_nedbor.content)))
        
        alpha = 0.5
        blended = alpha * norge_color + (1 - alpha) * nedbor_color

        return blended        
            
        

        
