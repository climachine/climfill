"""
DOCSTRING

    @author: Verena Bessenbacher
    @date: 23 04 2021
"""

import numpy as np
import xarray as xr
import xarray.ufuncs as xu
from interpolation import gapfill_interpolation, remove_ocean_points
from feature_engineering import  create_precip_binary, logscale_precip, create_lat_lon_features,
                                 create_time_feature, creade_embedded_features, stack_constant_maps
                                 normalise, stack
from clusterings import kmeans_clustering
from regression_learning import Imputation
from sklearn.ensemble import RandomForestRegressor
from postproc import exp_precip, renormalise, unstack

# load data
data = xr.open_dataset('/path/to/gappy/dataset')
constant_maps = xr.open_dataset('/path/to/constant/maps')

# create mask of missing values
mask = xu.isnan(data) 

# get list of variables
variables = data.coords['variables'].values

# step 1: interpolation
data = gapfill_interpolation(data)

# optional: save interpolation result for comparison

# optional: remove ocean points for reducing file size
landmask = xr.open_dataset('/path/to/landmask') # needs dims 'latitude' and 'longitude'
landlat, landlon = np.where(landmask)
data = remove_ocean_points(data, landlat, landlon)
mask = remove_ocean_points(mask, landlat, landlon)

# step 2: feature engineering

# (2): log-scale precipitation
ip = create_precip_binary(data, varname='tp')
data = data.to_dataset(dim='variable')
data['ip'] = ip
data = data.to_array()
data = logscale_precip(data, varname='tp')

# (2): add longitude and latitude as predictors
latitude_arr, longitude_arr = create_lat_lon_features(constant_maps)
constant_maps = constant_maps.to_dataset(dim='variable')
constant_maps['latdata'] = latitude_arr
constant_maps['londata'] = longitude_arr
constant_maps = constant_maps.to_array()

# (2): add time as predictor
time_arr = create_time_feature(data):
data = data.to_dataset(dim='variable')
data['timedat'] = time_arr
data = data.to_array()

# (2): add time lags as predictors
lag_7ff = create_embedded_features(data, s=0, l=7)
lag_7 = create_embedded_features(data, s=7, l=0)
lag_30 = create_embedded_features(data, s=30, l=7)
lag_180 = create_embedded_features(data, s=180, l=30)
data = xr.concat([data, lag_7ff, lag_7, lag_30, lag_180], dim='variable', join='left', fill_value=0)

# (2): concatenate constant maps and variables and features
constant_maps = stack_constant_maps(data, constant_maps)
data = xr.concat([data, constant_maps], dim='variable')

# intermediate step: prepare for regression learning
data, datamean, datastd = normalise(data)

# stack into tabular data
data = stack(data)
mask = stack(mask)

# step 3: clustering

# (3): clustering settings
epochs = np.arange(50,150,1)
random.seed(0)
epochs = random.choices(epochs, k=3) 

# (3) create clusters 
for e in epochs:
    logging.info(f'start epoch {e}...')
    labels = kmeans_clustering(data, nfolds=e)
    for f in range(e):
        logging.info(f'start fold {f}...')
        databatch = data[labels == f,:]
        maskbatch = mask[labels == f,:]

        # step 4: regression learning
        rf_settings = {'n_estimators': 100,
                      'min_samples_leaf': 2,
                      'max_features': 0.5, 
                      'max_samples': 0.5}
        regr_dict = {variable: RandomForestRegressor(**rf_settings) for variable in variables}
        maxiter = 10

        impute = Imputation(maxiter=maxiter)
        imputed_data, fitted_regr_dict = impute.impute(databatch, maskbatch, regr_dict)

        # imputed_data.to_netcdf ... 

# add cluster together again
data_imputed = data.copy(deep=True)
data_imputed = data_imputed.where(data_imputed == 0,0) # set all values zero

for e in epochs:
    filenames = glob.glob('path/to/files/of/this/epoch')
    data_epoch = xr.open_mfdataset(filenames, combine='nested', concat_dim='datapoints').load()

    data_epoch = data_epoch['data']
    data_epoch = unstack(data_epoch)
    data_epoch = data_epoch.sel(variable=variables)

    data_imputed = data_imputed + data_epoch

data_imputed = data_imputed / len(epochs)

# unstack
data_imputed = unstack(data_imputed

# renormalise
data_imputed = renormalise(data_imputed, datamean, datastd)

# save result
# data_imputed.to_netcdf(...)
