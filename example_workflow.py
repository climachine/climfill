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

This file shows exemplary a workflow that guides you through the four
steps of the CLIMFILL framework to gapfill your gridded geoscientific
dataset.
"""

import random

import numpy as np
import xarray as xr
import regionmask
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans

from climfill.feature_engineering import (
    create_embedded_feature,
    create_lat_lon_features,
    create_time_feature,
    stack_constant_maps,
)
from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging
from climfill.regression_learning import Imputation
from create_test_data import (
    create_constant_test_data,
    create_gappy_test_data,
)

# load your data
print("load data ...")
data = create_gappy_test_data()
constant_maps = create_constant_test_data()
landmask = regionmask.defined_regions.natural_earth.land_110.mask(data.lon,data.lat)
landmask = landmask.where(landmask!=0,1) # boolean mask: land is True, ocean is False 
landmask = landmask.where(~np.isnan(landmask),0)
landmask = landmask.astype(bool)

# get list of variables
print("get list of variables ...")
varnames = data.coords["variable"].values
print(varnames)

# step 1: interpolation
print("step 1: initial interpolation ...")

# step 1.1.: divide into monthly climatology and anomalies
data_monthly = data.groupby('time.month').mean()
data_anom = data.groupby('time.month') - data_monthly  

# step 1.2: gapfill monthly data with thin-plate-spline interpolation
rbf_kwargs = {'tp':    {'neighbors': 100, # cross-validate and consult Table A2 in publication
                        'smoothing': 0.1, 
                        'degree': 1},
              'swvl1': {'neighbors': 100,
                        'smoothing': 10, 
                        'degree': 2}, 
              'skt':   {'neighbors': 100,
                        'smoothing': 0.1,
                        'degree': 1},
              'tws':   {'neighbors': 100,
                        'smoothing': 0.1,
                        'degree': 1}}
data_monthly = gapfill_thin_plate_spline(data_monthly, landmask, rbf_kwargs)

# step 1.3: gapfill anomalies with kriging
kriging_kwargs = {'tp':    {'constant_value': 0.01, # cross-validate and consult Table A2 in publication
                            'length_scale': 1.0,
                            'repeats': 10,
                            'npoints': 100},
                  'swvl1': {'constant_value': 100.0,
                            'length_scale': 10.0,
                            'repeats': 1,
                            'npoints': 10},
                  'skt':   {'constant_value': 100.0,
                            'length_scale': 30.0,
                            'repeats': 10,
                            'npoints': 100},
                  'tws':   {'constant_value': 0.1,
                            'length_scale': 50.0,
                            'repeats': 5,
                            'npoints': 1000}}
#import warnings # DEBUG
#warnings.simplefilter('ignore')
data_anom = gapfill_kriging(data_anom, landmask, kriging_kwargs)

# step 1.4: add monthly climatology and anomalies back together
data = data_anom.groupby('time.month') + data_monthly
data = data.drop('month') # month not needed anymore

# necessary if full days are missing: fill all remaining gaps with variable mean
if np.isnan(data).sum() != 0: # if still missing values present
    print('still missing values treatment')
    variable_mean = data.mean(dim=("time", "lat", "lon"))
    data = data.fillna(variable_mean)

# step 2: feature engineering
print("step 2: feature engineering ...")

# step 2.1:  add longitude and latitude as predictors
latitude_arr, longitude_arr = create_lat_lon_features(constant_maps)
constant_maps = constant_maps.to_dataset(dim="variable")
constant_maps["latdata"] = latitude_arr
constant_maps["londata"] = longitude_arr
constant_maps = constant_maps.to_array()

# step 2.2: create mask of missing values
print("create mask of missing values ...")
mask = np.isnan(data)

# step 2.3 (optional): remove ocean points for reducing file size
landlat, landlon = np.where(landmask)
data = data.isel(lon=xr.DataArray(landlon, dims="landpoints"),
                 lat=xr.DataArray(landlat, dims="landpoints"))
mask = mask.isel(lon=xr.DataArray(landlon, dims="landpoints"),
                 lat=xr.DataArray(landlat, dims="landpoints"))
constant_maps = constant_maps.isel(lon=xr.DataArray(landlon, dims="landpoints"),
                                   lat=xr.DataArray(landlat, dims="landpoints"))

# step 2.4: add time as predictor
time_arr = create_time_feature(data)
data = data.to_dataset(dim="variable")
data["timedat"] = time_arr
data = data.to_array()

# step 2.5: add time lags as predictors
lag_007b = create_embedded_feature(data, start=-7,   end=0, name='lag_7b')
lag_030b = create_embedded_feature(data, start=-30,  end=-7, name='lag_30b')
lag_180b = create_embedded_feature(data, start=-180, end=-30, name='lag_180b')
lag_007f = create_embedded_feature(data, start=0,    end=7, name='lag_7f')
lag_030f = create_embedded_feature(data, start=7,    end=30, name='lag_30f')
lag_180f = create_embedded_feature(data, start=30,   end=180, name='lag_180f')
data = xr.concat(
    [data, lag_007b, lag_030b, lag_180b, lag_007f, lag_030f, lag_180f], 
    dim="variable", join="left", fill_value=0)

# fill still missing values at beginning of time series
varmeans = data.mean(dim=('time'))
data = data.fillna(varmeans)

# step 2.6: concatenate constant maps and variables and features
constant_maps = stack_constant_maps(data, constant_maps)
data = xr.concat([data, constant_maps], dim="variable")

# step 2.7: normalise data
datamean = data.mean(dim=("time", "landpoints"))
datastd = data.std(dim=("time", "landpoints"))
data = (data - datamean) / datastd

# step 2.8: stack into tabular data
data = data.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T
mask = mask.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T

# step 3: clustering
print("step 3: clustering ...")
data_imputed = xr.full_like(data.sel(variable=varnames).copy(deep=True), np.nan) # xr.full_like creates view
n_clusters = 30
labels = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, batch_size=1000, random_state=0).fit_predict(data)

# step 4: regression learning
print(f"step 4: regression learning ...")
for c in range(n_clusters):

    # step 4.1: select cluster
    print(f"cluster {c} ...")
    databatch = data[labels == c, :]
    maskbatch = mask[labels == c, :]
    idxs = np.where(labels == c)[0]

    rf_settings = {'n_estimators': 300, 
                   'min_samples_leaf': 2,
                   'max_features': 0.5, 
                   'max_samples': 0.5, 
                   'bootstrap': True,
                   'warm_start': False,
                   'n_jobs': 1, # depends on your number of cpus
                   'verbose': 0}
    regr_dict = {varname: RandomForestRegressor(**rf_settings) for varname in varnames}
    verbose = 1
    maxiter = 1

    impute = Imputation(maxiter=maxiter)
    # note that the following step takes quite long because the
    # toy dataset consists of random numbers and the RandomForest
    # has quite a hard time fitting this. Will be much faster with
    # correlated non-white noise data.
    databatch_imputed, regr_dict = impute.impute(
        databatch, maskbatch, regr_dict, verbose=2
    )

    databatch_imputed = databatch_imputed.sel(variable=varnames)

    data_imputed[idxs, : ] = databatch_imputed
    #break # DEBUG

# unstack
print("unstack ...")
data = data.set_index(datapoints=("time", "landpoints")).unstack("datapoints")

# renormalise
print("renormalise and exp precip ...")
data = data * datastd + datamean

# save result
print("save result ...")
# data_imputed.to_netcdf(...)
print("DONE")
