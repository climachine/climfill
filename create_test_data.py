import numpy as np
import pandas as pd
import xarray as xr
from sklearn.utils.random import sample_without_replacement

def create_test_data():

    year = '2003'
    time = pd.date_range('{}-01-01'.format(year),
                         freq="D", 
                         periods=365)
    lat = np.arange(-90,90,1)
    lon = np.arange(-180,180,1)
    variables = ['ground temperature','surface layer soil moisture','precipitation','terrestrial water storage']

    data = xr.DataArray(np.random.rand(len(time),len(lat),len(lon),len(variables)),
                        coords=[time,lat, lon, variables],
                        dims=['time','lat','lon','variables'])

    frac_missing = 0.5
    n_samples = data.size * frac_missing 
    idxs = sample_without_replacement(data.size, n_samples)
    data.values.flat[idxs] = np.nan

    return data
