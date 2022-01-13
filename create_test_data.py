import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from sklearn.utils.random import sample_without_replacement


def create_gappy_test_data():

    year = "2003"
    time = pd.date_range("{}-01-01".format(year), freq="D", periods=365)
    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)
    variables = [ # ERA5 variable names
        "skt", # ground temperature
        "swvl1", # surface layer soil moisture
        "tp", # precipitation
        "tws", # terrestrial water storage
    ]

    data = xr.DataArray(
        np.random.rand(len(variables), len(time), len(lat), len(lon)),
        coords=[variables, time, lat, lon],
        dims=["variable", "time", "lat", "lon"],
    )

    frac_missing = 0.1
    n_samples = data.size * frac_missing
    idxs = sample_without_replacement(data.size, n_samples)
    data.values.flat[idxs] = np.nan

    return data


def create_constant_test_data():

    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)
    variables = ["topography", "land cover"]

    data = xr.DataArray(
        np.random.rand(len(variables), len(lat), len(lon)),
        coords=[variables, lat, lon],
        dims=["variable", "lat", "lon"],
    )
    return data
