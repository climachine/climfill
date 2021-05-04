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

This file gives you convenience functions for clustering the data
in environmentally similar points. At the end of the file, an 
example workflow is shown.
"""

# parameter import
from sklearn.cluster import MiniBatchKMeans, OPTICS
import numpy as np
import random
import xarray as xr
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

def random_clustering(data, nfolds):
    """
    Cluster data into random clusters 

    Parameters
    ----------
    data: xarray dataarray, with features as columns and datapoints as rows

    nfolds: number of clusters, for more details see 
        sklearn.cluster.MiniBatchKMeans

    Returns
    ----------
    labels: for each datapoint, the label of which cluster it belongs to
    """
    labels = np.full(data.shape[0], 0)
    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs) # in place
    for s, split in enumerate(np.array_split(idxs, nfolds)):
        labels[split] = s
    return labels

def kmeans_clustering(data, nfolds, init='k-means++'):
    """
    Cluster data into environmentally similar datapoints by using 
    scikit.learn MiniBatchKMeans algorithm. 

    Parameters
    ----------
    data: xarray dataarray, with features as columns and datapoints as rows

    nfolds: number of clusters, for more details see 
        sklearn.cluster.MiniBatchKMeans

    init: initial cluster centers, for more details see 
        sklearn.cluster.MiniBatchKMeans

    Returns
    ----------
    labels: for each datapoint, the label of which cluster it belongs to, for 
        more details see sklearn.cluster.MiniBatchKMeans
    """
    labels = MiniBatchKMeans(n_clusters=nfolds, verbose=0, batch_size=1000, 
                             random_state=0).fit_predict(data)
    return labels

def dbscan_clustering(data, nfolds):
    """
    Cluster data into clusters with sklearn.cluster.OPTICS 
    (non-linear clustering)

    Parameters
    ----------
    data: xarray dataarray, with features as columns and datapoints as rows

    nfolds: number of clusters, for more details see sklearn.cluster.OPTICS

    Returns
    ----------
    labels: for each datapoint, the label of which cluster it belongs to, for 
        more details see sklearn.cluster.OPTICS
    """
    clustering = OPTICS(max_eps=1, cluster_method='dbscan').fit(data)
    return clustering.labels_
