"""
impute given dataset with randomforest in random bags of data

    @author: Verena Bessenbacher
    @date: 06 07 2020
"""

# parameter import
from namelist import varnames
from sklearn.cluster import MiniBatchKMeans, OPTICS
import argparse
import numpy as np
import random
from calendar import monthrange
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--missingness', '-m', dest='missingness', type=str, default='real')
    parser.add_argument('--frac_missing', '-p', dest='frac_missing', type=float, default=None)
    parser.add_argument('--debug', dest='idebugspace', action='store_true')
    parser.add_argument('--no-debug', dest='idebugspace', action='store_false')
    parser.set_defaults(idebugspace=True)
    args = parser.parse_args()

    # data settings
    missingness = args.missingness
    frac_missing = args.frac_missing
    idebugspace = args.idebugspace
    icrossval = False
    if icrossval:
        from namelist import crossvalpath
        largefilepath = crossvalpath
    else:
        from namelist import largefilepath

    # method settings
    bagging_method = 'kmeans'
    if idebugspace:
        epochs = np.arange(20,60,1)
    else:
        epochs = np.arange(50,150,1)
    random.seed(0)
    epochs = random.choices(epochs, k=3) 

    # read feature table
    logging.info(f'read feature table...')
    # ATTENTION THESE ARE NOT YET THE SAME
    # differences larger than float point uncertainties in tp and swvl1, smaller in skt, none at all in tws
    # for all timesteps (seasonality pattern) and especially places on earth with high missingness
    # idea: something wrong / different in interpolation function? since now temporal and variable mean are calculated before any gapfilling happens
    # import IPython; IPython.embed()
    # OLD
    #init_impute = 'smart'
    #data = xr.open_dataset(largefilepath + f'features_init_{missingness}_{frac_missing}_{init_impute}_idebug_{idebugspace}.nc').load()
    #lostmask = xr.open_dataset(largefilepath + f'lostmask_init_{missingness}_{frac_missing}_{init_impute}_idebug_{idebugspace}.nc').load()
    #data = data['__xarray_dataarray_variable__'].T
    #logging.info(f'preproc shizzl...')
    #lostmask = lostmask['tp'].T
    #lostmask = lostmask.stack(datapoints=('time','landpoints')).T
    # NEW
    data = xr.open_dataset(largefilepath + f'features_init_{missingness}_{frac_missing}_idebug_{idebugspace}.nc')
    lostmask = xr.open_dataset(largefilepath + f'lostmask_init_{missingness}_{frac_missing}_idebug_{idebugspace}.nc')
    data = data['data'] # netcdf can only save datasets, we use only dataarrays because of more convenient indexing options
    lostmask = lostmask['data']
    #import IPython; IPython.embed()
    
    # create clusters and save
    for e in epochs:
        logging.info(f'start epoch {e}...')
        labels = kmeans_clustering(data, nfolds=e)
        for f in range(e):
            logging.info(f'start fold {f}...')
            batch = data[labels == f,:]
            batchlost = lostmask[labels == f,:]

            batch.to_dataset(name='data').to_netcdf(largefilepath + f'features_init_label_e{e}f{f}_{bagging_method}_{missingness}_{frac_missing}_idebug_{idebugspace}.nc')
            batchlost.to_dataset(name='data').to_netcdf(largefilepath + f'lostmask_init_label_e{e}f{f}_{bagging_method}_{missingness}_{frac_missing}_idebug_{idebugspace}.nc', encoding={'data':{'dtype':'bool'}}) # needs explicit bool otherwise lostmask is saved as int (0,1) and numpy selects all datapoints as missing in imputethis since 0 and 1 are treated as true and no false are found
