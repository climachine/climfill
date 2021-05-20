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
import xarray.ufuncs as xu
from sklearn.ensemble import RandomForestRegressor

from climfill.clustering import kmeans_clustering
from climfill.feature_engineering import (
    create_embedded_features,
    create_lat_lon_features,
    create_precip_binary,
    create_time_feature,
    logscale_precip,
    normalise,
    stack,
    stack_constant_maps,
)
from climfill.interpolation import gapfill_interpolation, remove_ocean_points
from climfill.postproc import exp_precip, renormalise, unstack
from climfill.regression_learning import Imputation
from create_test_data import (
    create_constant_test_data,
    create_gappy_test_data,
    create_landmask,
)

# load data
print("load data ...")
data = create_gappy_test_data()
constant_maps = create_constant_test_data()
landmask = create_landmask()

# create mask of missing values
print("create mask of missing values ...")
mask = xu.isnan(data)

# get list of variables
print("get list of variables ...")
varnames = data.coords["variable"].values

# step 1: interpolation
print("step 1: initial interpolation ...")
data = gapfill_interpolation(data)

# step 2: feature engineering
print("step 2: feature engineering ...")

# (2): log-scale precipitation
ip = create_precip_binary(data, varname="precipitation")
data = data.to_dataset(dim="variable")
data["ip"] = ip
data = data.to_array()
data = logscale_precip(data, varname="precipitation")

# (2): add longitude and latitude as predictors
latitude_arr, longitude_arr = create_lat_lon_features(constant_maps)
constant_maps = constant_maps.to_dataset(dim="variable")
constant_maps["latdata"] = latitude_arr
constant_maps["londata"] = longitude_arr
constant_maps = constant_maps.to_array()

# optional: remove ocean points for reducing file size
# landmask needs dimensions with names 'latitude' and 'longitude'
# landmask = xr.open_dataset("/path/to/landmask")
data = remove_ocean_points(data, landmask)
mask = remove_ocean_points(mask, landmask)
constant_maps = remove_ocean_points(constant_maps, landmask)

# (2): add time as predictor
time_arr = create_time_feature(data)
data = data.to_dataset(dim="variable")
data["timedat"] = time_arr
data = data.to_array()

# (2): add time lags as predictors
lag_7ff = create_embedded_features(data, varnames, window_size=0, lag=7)
lag_7 = create_embedded_features(data, varnames, window_size=7, lag=0)
lag_30 = create_embedded_features(data, varnames, window_size=30, lag=7)
lag_180 = create_embedded_features(data, varnames, window_size=180, lag=30)
data = xr.concat(
    [data, lag_7ff, lag_7, lag_30, lag_180], dim="variable", join="left", fill_value=0
)

# (2): concatenate constant maps and variables and features
constant_maps = stack_constant_maps(data, constant_maps)
data = xr.concat([data, constant_maps], dim="variable")

# intermediate step: prepare for regression learning
data, datamean, datastd = normalise(data)

# stack into tabular data
data = stack(data)
mask = stack(mask)

# step 3: clustering

# (3): clustering settings
n_epochs = 3
epochs = np.arange(5)
random.seed(0)
epochs = random.choices(epochs, k=n_epochs)

# (3) create clusters
print("step 3: clustering ...")
data_imputed = xr.full_like(data, np.nan)
data_imputed = data_imputed.expand_dims({"epochs": n_epochs}, axis=0)
data_imputed = data_imputed.copy(deep=True)  # xr.full_like creates view

for e, epoch in enumerate(epochs):

    # clustering data for this epoch
    labels = kmeans_clustering(data, nfolds=epoch)

    for f in range(epoch):

        # select data chunk in cluster
        databatch = data[labels == f, :]
        maskbatch = mask[labels == f, :]
        idxs = np.where(labels == f)[0]

        # step 4: regression learning
        print(f"step 4: regression learning for epoch {epoch} cluster {f} ...")
        rf_settings = {
            "n_estimators": 100,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "max_samples": 0.5,
        }

        regr_dict = {
            variable: RandomForestRegressor(**rf_settings) for variable in varnames
        }

        maxiter = 1

        impute = Imputation(maxiter=maxiter)
        databatch_imputed, regr_dict = impute.impute(
            databatch, maskbatch, regr_dict, verbose=1
        )

        data_imputed[e, idxs, :] = databatch_imputed

# take mean through all epochs
data_imputed = data_imputed.mean(dim="epochs")

# unstack
data_imputed = unstack(data_imputed)

# renormalise
data_imputed = renormalise(data_imputed, datamean, datastd)

# save result
# data_imputed.to_netcdf(...)
