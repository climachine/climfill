import numpy as np
import pandas as pd
import xarray as xr
from sklearn.utils.random import sample_without_replacement


def create_gappy_test_data():

    year = "2003"
    time = pd.date_range("{}-01-01".format(year), freq="D", periods=365)
    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)
    variables = [
        "ground temperature",
        "surface layer soil moisture",
        "precipitation",
        "terrestrial water storage",
    ]

    data = xr.DataArray(
        np.random.rand(len(variables), len(time), len(lat), len(lon)),
        coords=[variables, time, lat, lon],
        dims=["variable", "time", "latitude", "longitude"],
    )

    frac_missing = 0.5
    n_samples = data.size * frac_missing
    idxs = sample_without_replacement(data.size, n_samples)
    data.values.flat[idxs] = np.nan

    return data


def create_constant_test_data():

    year = "2003"
    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)
    variables = ["topography", "land cover"]

    data = xr.DataArray(
        np.random.rand(len(variables), len(lat), len(lon)),
        coords=[variables, lat, lon],
        dims=["variable", "latitude", "longitude"],
    )
    return data


def create_landmask():  # TODO from regionmask

    year = "2003"
    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)

    data = np.ones((len(lat), len(lon)))
    data[:10, :10] = 0
    data = data.astype(bool)

    data = xr.DataArray(data, coords=[lat, lon], dims=["latitude", "longitude"])
    return data
