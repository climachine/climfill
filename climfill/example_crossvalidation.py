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

This file shows exemplary a workflow that guides you through the cross-
validation necessary to find the optimal parameters for interpolation 
(kriging and thin-plate-spline) as well as for the Random Forest function. 
The Cross-validation happens on minicubes of deleted, originally observed
data in a year of the users choice.
"""

import numpy as np

from create_test_data import create_constant_test_data

from climfill.verification import delete_minicubes

#  load your data
print("Load example data ...")
data = create_gappy_test_data()

# create mask of missing values: before first gapfill
print("create mask of missing values...")
mask = np.isnan(data)

# create additional missing values for verification/cross-validation (optional)
#data.to_netcdf('data_orig... # save original values of minicubes for verification
print("create additonal missing values for CV...")
frac_mis = 0.1 
ncubes = 20
crossvalidation_year = '2003'
for varname in varnames:
    tmp = delete_minicubes(mask.sel(time=crossvalidation_year, variable=varname).drop('variable').load(),
                           frac_mis, ncubes)
    mask.loc[dict(variable=varname, time=crossvalidation_year)] = tmp
data = data.where(np.logical_not(mask))

# perform your cross-validation here..

