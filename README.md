# CLIMFILL : A Framework for Intelligently Gap-filling Earth Observations

CLIMFILL fills gaps in gridded geoscientific observational data by taking into account both spatiotemporally neighboring points and multivariate dependencies. It takes a multivariate dataset with any number and pattern of missing values per variable and returns the dataset with all missing points replaced by estimates. CLIMFILL is a framework, not a clearly defined method. Therefore, for each of the gap-filling steps taken, the user needs to define a method that suits their particular needs best. Some of the most common functions are however already part of the package. For a full description of the framework, see [1].

CLIMFILL consists of four steps:

1. Interpolation: initially, the missing values are filled by an interpolation using spatiotemporally close points. In [1], this is done by running a 5x5x5 3-dimensional convolutional median filter from `scipy.ndimage.filters` for each variable. However, any interpolation method (bilinear, cubic, kriging, ...) can be used here.
2. Feature engineering: In the spirit of data science, in the second step descriptive features are created depending on the individual needs of the data. For example, time, and space, expressed in latitude and longitude, can be features. Furthermore, running means of important variables can be features to inform about slowly changing processes. Helper functions to create such features are provided.
3. Clustering: Before the last step, the data is divided into environmentally similar points accross space and time. Any classification method can be used here, in the package `sklearn.cluster.KMeans`, `sklearn.cluster.OPTICS` and random clustering are supported.
4. Regression learning: the initial gap-fill estimates from step 1 are iteratively updated by learning and applying any regression function to the data using a method adapted from the MissForest Algorithm [2]. In the `example_workflow.py`, `sklearn.ensemble.RandomForestRegressor` is used.

The necessary functions are in the respective scripts. An example workflow going through all four steps is shown in `example_workflow.py`.

## Installation

`pip install git+https://github.com/climachine/climfill`

## Documentation
For a full documentation, check [1]. Furthermore, each function and class has a doc string.

## Cautionary Notes
- Depending on the amount of the data, sometimes it is useful to save intermediate results and calculate the most expensive Step 4 for each cluster separately. This can reduce memory and CPU usage. The functions are written as such that they can easily be parallelised over the clusters or epochs.
- The package is very much work in progress. Any feedback is highly appreciated.

## License
The work is distributed under the Apache-2.0 License.

## References
- [1] Bessenbacher, V., Gudmundsson, L. and Seneviratne, S.I.: CLIMFILL: A Framework for Intelligently Gap-filling Earth Observations (submitted to Geoscientific Model Development on 20th May 2021) 
and references therein, especially
- [2] Stekhoven, D. J. and Buehlmann, P. (2012): MissForest -- non-parametric missing value imputation for mixed-type data. Bioinformatics, 28, 1, 112-118.
