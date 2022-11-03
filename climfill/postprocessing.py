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

This file gives you teh convenience function for bringing the data
back to its lat lon time shape.
"""

import numpy as np
import xarray as xr

def to_latlon(data, landmask): 
    """
    Convert 3-dimensional data (variable, landpoints, time) that usually exists
    after gapfilling and unstacking, back to its 4-dimensional shape (variable,
    lat, lon, time) for analysis and final result.

    Parameters
    ----------
    data: xarray dataarray, with dimensions variable, landpoints, time

    landmask: xarray dataarray, boolean, with dimensions lat and lon, where
          values that have been gapfilled (e.g. on land) are set True and
          values that have been ignored for gapfilling (e.g. in the ocean) are
          set to False.

    Returns
    ----------
    tmp: xarray with dimensions variable and time from data and lat, lon from
         landmask, where the landpoints are sorted back to their original
         geographic location
    """
    shape = landmask.shape
    landlat, landlon = np.where(landmask)
    tmp = xr.DataArray(np.full((data.coords['variable'].size,
                                data.coords['time'].size,
                                shape[0],shape[1]),np.nan), 
                       coords=[data.coords['variable'], data.coords['time'], 
                               landmask.coords['lat'], landmask.coords['lon']], 
                       dims=['variable','time','lat','lon'])
    tmp.values[:,:,landlat,landlon] = data
    return tmp
