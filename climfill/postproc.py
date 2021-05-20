"""
Copyright 2021 ETH Zurich, contributor: Verena Bessenbacher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file gives you convenience functions for concatenating the data
back together from the individual clusters and datasets and bring it
back to its lat lon time shape.
"""

import numpy as np
import xarray.ufuncs as xu


def unstack(data):
    return data.set_index(datapoints=("time", "landpoints")).unstack("datapoints")


def renormalise(data, datamean, datastd):
    data = data * datastd + datamean
    return data


def exp_precip(data):
    # all -20 get nan
    data.loc["tp"] = data.loc["tp"].where(data.loc["tp"] > -19, np.nan)
    # all nan stay nan
    data.loc["tp"] = xu.exp(data.loc["tp"])
    # all nan get zero precip
    data.loc["tp"] = data.loc["tp"].where(~np.isnan(data.loc["tp"]), 0)
    return data
