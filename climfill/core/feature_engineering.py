"""
Copyright 2021 ETH Zurich, contributor: Verena Bessenbacher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file gives you the functions for adding descriptive features to
your dataset prior regression learning, most importantly the ability
to add embedded features that describe slowly changing, important
processes in your dataset.
"""

import logging

# imports
import sys

import numpy as np
import xarray as xr
import xarray.ufuncs as xu

logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
from scipy.ndimage.filters import generic_filter


def log_fracmis(data, logtext=''):
    frac_mis = (np.isnan(data).sum() / data.size).values
    logging.info(f'fraction missing {logtext}: {frac_mis}')

def logscale_precip(data, varname='tp'):
    """
    log-scale variable "varname" in data

    Parameters
    ----------
    data: xarray dataarray, where varname is the variable that needs to be 
        log-scaled

    Returns
    ----------
    data: data with variable log-scaled
    """
    # define lower threshold for "no rain"
    data.loc[varname] = data.loc[varname].where(data.loc[varname] > 0.000001,
                                                -1) 
    data.loc[varname] = xu.log(data.loc[varname])

    # all zero precip got -1, all -1 got nan, all nan get -20
    data.loc[varname] = data.loc[varname].where(~np.isnan(data.loc[varname]),
                                                -20) 
    return data

def create_precip_binary(data, varname='tp'):
    ip = (data.loc[varname] < 0.00001)*1
    ip = ip.rename('ip')
    return ip

def normalise(data):
    """
    for each variable (i.e. across the dimension 'variable') normalise the data 
        to mean zero and standard deviation one

    Parameters
    ----------
    data: xarray dataarray, with dimensions variable, time, landpoints

    Returns
    ----------
    data: data normalised
    datamean: mean for each variable, for renormalisation
    datastd: std for each variable, for renormalisation
    """
    datamean = data.mean(dim=('time','landpoints'))
    datastd = data.std(dim=('time','landpoints'))
    data = (data - datamean) / datastd
    return data, datamean, datastd

def stack(data):
    # add select variable to remove it in gapfill?
    return data.stack(datapoints=('time','landpoints'))
               .reset_index('datapoints').T.to_dataset(name='data')

def create_lat_lon_features(constant_maps):
    """
    create latitude and longitude as additional feature for data

    Parameters
    ----------
    data: xarray dataarray, with dimensions including latitude and longitude

    Returns
    ----------
    latitude_arr
    longitude_arr
    """
    londata, latdata = np.meshgrid(constant_maps.longitude, 
                                   constant_maps.latitude)
    latitude_arr = (('latitude', 'longitude'), latdata)
    longitude_arr = (('latitude', 'longitude'), londata)
    return latitude_arr, longitude_arr

def create_time_feature(data):
    """
    create timestep as additional feature for data

    Parameters
    ----------
    data: xarray dataarray, with dimensions including landpoints, time 

    Returns
    ----------
    time_arr: xarray with same dimensions as one feature in array describing 
        time step
    """
    _, ntimesteps, nlandpts = data.shape
    timedat = np.arange(ntimesteps)
    timedat = np.tile(timedat, nlandpoints).reshape(nlandpts,*timedat.shape).T
    time_arr = (('time','landpoints'), timedat)
    return time_arr

def create_embedded_features(data, s, l, varnames):
    """
    for each variable, create embedded features of data with mean over window 
        size s and time lag l

    Parameters
    ----------
    data: xarray dataarray, with dimensions including variable, time 
    varnames: list of all variables for calculating this embedded feature
    s: int, window size in days
    l: int, lag of window from today in days

    Returns
    ----------
    tmp: embedded features of variables to be added to data
    """

    # rolling window average
    tmp = data.sel(variable=varnames)
              .rolling(time=l-s, center=False, min_periods=1).mean()

    # overwrite time stamp to current day
    tmp = tmp.assign_coords(time=[time + np.timedelta64(l,'D') 
                                  for time in tmp.coords['time'].values])

    # rename feature to not overwrite variable
    tmp = tmp.assign_coords(variable=[f'{var}lag_{s}ff' for var in varnames])

    # fill missing values in lagged features at beginning or end of time series
    varmeans = tmp.mean(dim=('time'))
    tmp = tmp.fillna(varmeans)

    return tmp

def stack_constant_maps(data, constant_maps):
    ntimesteps = data.coords['time'].size
    constant_maps = np.repeat(constant_maps, ntimesteps, axis=1)
    constant_maps['time'] = data['time'] # needs timestep for concat to work
    return constant_maps
