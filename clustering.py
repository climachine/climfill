"""
impute given dataset with randomforest in random bags of data

    @author: Verena Bessenbacher
    @date: 06 07 2020
"""

# parameter import
from sklearn.cluster import MiniBatchKMeans, OPTICS
import numpy as np
import random
import xarray as xr
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

def random_clustering(data, nfolds):
    labels = np.full(data.shape[0], 0)
    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs) # in place
    for s, split in enumerate(np.array_split(idxs, nfolds)):
        labels[split] = s
    return labels

def kmeans_clustering(data, nfolds, init='k-means++'):
    labels = MiniBatchKMeans(n_clusters=nfolds, verbose=0, batch_size=1000, random_state=0).fit_predict(data)
    return labels

def dbscan_clustering(data, nfolds):
    clustering = OPTICS(max_eps=1, cluster_method='dbscan').fit(data)
    return clustering.labels_

if __name__ == '__main__':

    # method settings
    epochs = np.arange(50,150,1)
    random.seed(0)
    epochs = random.choices(epochs, k=3) 

    # read feature table
    data = xr.open_dataset('/path/to/gappy/dataset')
    mask = xr.open_dataset('/path/to/mask/dataset')
    data = data['data'] # netcdf can only save datasets, we use only dataarrays because of more convenient indexing options
    mask = mask['data']
    
    # create clusters and save
    for e in epochs:
        logging.info(f'start epoch {e}...')
        labels = kmeans_clustering(data, nfolds=e)
        for f in range(e):
            logging.info(f'start fold {f}...')
            databatch = data[labels == f,:]
            maskbatch = mask[labels == f,:]

            # databatch.to_netcdf ...
            # maskbatch.to_netcdf ...
