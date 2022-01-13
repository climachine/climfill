"""
Copyright 2021 ETH Zurich, author: Verena Bessenbacher

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

import numpy as np
import xarray.ufuncs as xu


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
    londata, latdata = np.meshgrid(constant_maps.lon, constant_maps.lat)
    latitude_arr = (("latitude", "longitude"), latdata)
    longitude_arr = (("latitude", "longitude"), londata)
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
    timedat = np.tile(timedat, nlandpts).reshape(nlandpts, *timedat.shape).T
    time_arr = (("time", "landpoints"), timedat)
    return time_arr


def create_embedded_feature(data, start=-7, end=0, name='lag_7b'):
    """
    create moving window mean along time axis from day 'start' until
    day 'end' relative to current day using xr.DataArray.rolling

    Parameters
    ----------
    data: xarray dataarray, with dimensions including variable, time

    start: int, start of moving average in days from current day

    end: int, end of moving average in days from current day

    name: name of the resulting variable in the returned data

    Returns
    ----------
    feature: embedded features of variables to be added to data
    """

    # TODO use xr.Dataarray.shift

    varnames = data.coords["variable"].values

    length = np.abs(start - end)
    offset = max(start*(-1),end*(-1))
    feature = data.rolling(time=length, center=False, min_periods=1).mean()
    feature = feature.assign_coords(time=[time + np.timedelta64(offset,'D') for time in feature.coords['time'].values])
    feature = feature.assign_coords(variable=[f'{var}{name}' for var in varnames])

    return feature

def stack_constant_maps(data, constant_maps):
    ntimesteps = data.coords["time"].size
    constant_maps = constant_maps.expand_dims({"time": ntimesteps}, axis=1)
    # constant_maps = np.repeat(constant_maps, ntimesteps, axis=1)
    constant_maps["time"] = data["time"]
    return constant_maps
