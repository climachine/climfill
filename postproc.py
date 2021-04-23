"""
version on daily resolution, with dask if possible

    @author: Verena Bessenbacher
    @date: 20 03 2020
"""

import sys
import glob
import xarray as xr
import logging
import argparse
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # no matplotlib logging
from namelist import largefilepath, varnames
import random
import numpy as np
import xarray.ufuncs as xu

def unstack(data):
    return data.set_index(datapoints=('time','landpoints')).unstack('datapoints')

def renormalise(data, datamean, datastd):
    data = data * datastd + datamean
    return data

def exp_precip(data):
    data.loc['tp'] = data.loc['tp'].where(data.loc['tp'] > -19, np.nan) # all -20 get nan
    data.loc['tp'] = xu.exp(data.loc['tp']) # all nan stay nan
    data.loc['tp'] = data.loc['tp'].where(~np.isnan(data.loc['tp']), 0) # all nan get zero precip
    return data

if __name__ == '__main__':

    epochs = np.arange(50,150,1)
    random.seed(0)
    epochs = random.choices(epochs, k=3)

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


