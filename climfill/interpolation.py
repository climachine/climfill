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

This file gives you the function for filling any missing points
with a spatial interpolation (interpolation step) 
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

def gapfill_thin_plate_spline(data_monthly, landmask, rbf_kwargs):
    """
    Impute (i.e. Gapfill) data by applying thin-plate-spline interpolation
    on each variable and each timestep independently.

    Parameters
    ----------
    data_monthly: xarray dataarray, with coordinates time, lat, lon
        and variable

    landmask: xarray dataarray, boolean, where land is True and ocean is
        False

    rbf_kwargs: dict, with the keys being the names of all the variables
        in the data and the values dictionaries specifiying the parameter
        settings for the scipy.interpolate.RBFInterpolator

    Returns
    ----------
    imputed_data: xarray dataarray, data of the same shape as input data,
        where all values that were not missing are still the same and all
        values that were originally missing are imputed 
    """

    data = xr.full_like(data_monthly, np.nan)
    varnames = data.coords["variable"].values
    xx, yy = np.meshgrid(data.lon, data.lat)

    for month in data.month:
        for varname in varnames:

            # select month and variable
            tmp = data.sel(month=month, variable=varname)
            missing = landmask & np.isnan(tmp)
            notna = tmp.notnull()

            # check if some values are missing
            if missing.sum().values.item() == 0:
                continue
            print(month, varname)

            # select only missing points
            xy_obs = np.c_[xx[notna.values], yy[notna.values]] 
            xy_mis = np.c_[xx[missing.values], yy[missing.values]] 
            y_obs = tmp.values[notna]

            # interpolate missing values
            interpolator = RBFInterpolator(xy_obs, y_obs, **rbf_kwargs[varname])
            try:
                res = interpolator(xy_mis)
            except Exception as e:
                print(f'{month.item()}, {varname}, {e} in RBFInterpolator occurred, gaps not filled')
                res = np.full_like(xy_mis[:,0], np.nan)

            # save result
            data.loc[varname, month,:,:].values[missing] = res
    
    return data

def gapfill_kriging(data_anom, landmask, kriging_kwargs):
    """
    Impute (i.e. Gapfill) data by applying spatial kriging
    on each variable and each timestep independently.

    Parameters
    ----------
    data_monthly: xarray dataarray, with coordinates time, lat, lon
        and variable

    landmask: xarray dataarray, boolean, where land is True and ocean is
        False

    kriging_kwrags: dict, with the keys being the names of all the variables
        in the data and the values also being dictionaries specifiying the 
        parameter settings for the sklearn.gaussian_process.GaussianProcessRegressor
        as well as two keys specifying the number of repeats ('repeats')
        and the number of points considered ('npoints') for each fitting.

    Returns
    ----------
    imputed_data: xarray dataarray, data of the same shape as input data,
        where all values that were not missing are still the same and all
        values that were originally missing are imputed
    """

    varnames = data_anom.coords["variable"].values
    xx, yy = np.meshgrid(data_anom.lon, data_anom.lat)

    for day in data_anom.time:#[:2]: # DEBUG not to take too long
        for varname in varnames:

            # select day and variable
            tmp = data_anom.sel(time=day, variable=varname)
            missing = landmask & np.isnan(tmp)
            y = tmp.values[~np.isnan(tmp.values)]

            # select only missing values
            xy_test = np.c_[xx[missing.values], yy[missing.values]]
            xy_train = np.c_[xx[~np.isnan(tmp.values)], yy[~np.isnan(tmp.values)], y]
            if xy_test.size == 0:
                print(f'{day.values} {varname} SKIPPED no missing points')
                continue
        
            # gapfill missing values
            k, n = kriging_kwargs[varname]['repeats'], kriging_kwargs[varname]['npoints']
            constant_value = kriging_kwargs[varname]['constant_value']
            length_scale = kriging_kwargs[varname]['length_scale']
            res = np.full((k,xy_test.shape[0]), np.nan)

            for i in range(k):
                bounds = (1e-6,1e6)
                np.random.shuffle(xy_train)
                kernel = C(constant_value) * RBF(length_scale)
                gp = GaussianProcessRegressor(kernel, copy_X_train=False)
                try:
                    gp.fit(xy_train[:n,:2],xy_train[:n,-1])
                except ValueError as e:
                    print(f'{e} {varname} SKIPPED')
                Z = gp.predict(xy_test)
                res[i,:] = Z
            
            # save result 
            data_anom.loc[varname, day, :,:].values[missing.values] = res.mean(axis=0)

    return data_anom
