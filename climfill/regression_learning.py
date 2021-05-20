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

This file gives you the class for updating an initial guess of a
missing value in a dataset by taking into account the multivariate
dependence structure of the data.
"""

import copy

import numpy as np
from numpy import Inf


class Imputation:
    """
    A Gapfill procedure for multivariate data.

    An algorithm designed to fill gaps in tabular data by iteratively learning
    to regress one variable (or feature) with all the others. The algorithm
    needs initial estimates provided for the gaps which it uses as a starting
    point for the iterative procedure. Built upon [1] but with several
    adaptations that tailor it to the needs of geoscientific datasets (for more
    details see [2]).

    ...

    Parameters
    ----------
    epsilon: int, default=1e-5
        convergence criterion for the iterative algorithm. The difference in
        the data between two iterations is defined as the summed absolute
        difference between the data matrix of the new and old iteration and
        called delta. If the difference of the delta from the last iteration
        and the delta from the current one is below epsilon, the algorithm
        converged and the gapfill results are returned. This is a relaxed
        version of the convergence criterium in Stekhoven and
        Buehlmann (2012) [1].

    maxiter: int, default=Inf
        second convergence criterium for the iterative algorithm.
        Alternatively, even if epsilon is not reached, the algorithm stops
        after the maximum number of allowed iterations maxiter is reached and
        the gapfilled data is returned.

    miniter: int, default=5
        minimum number of iterations before convergence criterion is checked

    miniter_below: int, default=5
        minimum number of iterations after convergence criterion is checked
        before iteration is stopped. Only applied if convergence is reached via
        epsilon, not via maxiter

    References
    ----------
    .. [1] Stekhoven, D. J. and Buehlmann, P. (2012): MissForest -- non-
        parametric missing value imputation for mixed-type data.
        Bioinformatics, 28, 1, 112-118.
    .. [2] Bessenbacher, V., Gudmundsson, L., Seneviratne, S. I. (2021):
        CLIMFILL: A Framework for Intelligently Gap-filling Earth Observations
        (in prep.)

    Examples
    --------
    # TODO missing: init gapfill, data production

    from regression_learning import Imputation
    import xarray as xr
    import numpy as np

    data = xr.open_dataset('/path/to/gappy/dataset')
    mask = np.isnan(data)

    variables = ['temperature', 'precipitation']
    rf_settings = {'n_estimators': 100,
                  'min_samples_leaf': 2,
                  'max_features': 0.5,
                  'max_samples': 0.5}
    regr_dict = {variable: RandomForestRegressor(**kwargs)
                 for variable in variables}
    maxiter = 10

    impute = Imputation(maxiter=maxiter)
    imputed_data, fitted_regr_dict = impute.impute(data, mask, regr_dict)

    """

    def __init__(self, epsilon=1e-5, maxiter=Inf, miniter=5, miniter_below=5):

        self.epsilon = epsilon
        self.maxiter = maxiter  # maximal number of total iteration
        self.miniter = miniter  # minimal number of total iterations
        self.miniter_below = miniter_below  # minimal number of iterations
        # where epsilon is below threshold

    def _logiter(self, logtrunc):
        """
        Internal function to handle iteration log
        """
        logging.info(
            (
                f"{logtrunc} new delta: {np.round(self.delta_n, 9)}",
                f"diff: {np.round(self.delta_n_old - self.delta_n, 9)}",
                f"niter: {self.iter} niterbelow: {self.iter_below}",
            )
        )

    def impute(self, data, mask, regr_dict, kwargs={}, verbose=1, logtrunc=""):
        """
        Impute (i.e. Gapfill) data by regressing each variable (i.e. column)
        in data with all other variables in data.

        Parameters
        ----------
        data: xarray dataarray, feature table with rows as datapoints and
            columns as features

        mask:
            boolean xarray in the same shape as data, indicating which values
            were originally missing as True and all others as False

        regr_dict: dictionary where the keys are the names of the gappy
            variables that form a part of the columns in data. variable names
            must match the coordinate names of data column names. Values are
            instances of regression functions (e.g. scikit-learn's
            RandomForestRegressor) that have a fit() and a predict() method.

        kwargs: optional. Some regression functions need additional keywords
            for the .fit() method. provide these here

        verbose: optional, int debug verbosity

        Returns
        ----------
        imputed_data: data of the same shape as input data, where all values
            that were not missing are still the same and all values that were
            originally missing are imputed via regression learning

        fitted_regr_dict: dictionary where the variable names are the keys and
            the fitted regression functions (including all weights) per
            variable are the values
        """

        if mask.sum().values == 0:
            raise ValueError("mask does not have any True entry. abort")

        # define convergence criteria
        self.delta_n = Inf

        # set initial values for iteration params
        self.iter = 0
        self.iter_below = 0

        self.fittedregr = dict()

        # within a single run, without changing the dict, the order is not
        # mutated, see https://stackoverflow.com/questions/52268262/does-
        # pythons-dict-items-always-return-the-same-order
        varnames = [varname for varname, regr in regr_dict.items()]
        print(varnames)

        # while no convergence, loop over features and impute iteratively
        while True:

            # store previously imputed matrix
            data_old = data.copy(deep=True)
            # keep next line otherwise on euler diff gets zero. bug 05032021
            np.array_equal(data, data_old)
            for varname, Regr_orig in regr_dict.items():

                # this is very important. some concurrent.futures threads abort
                # because Regr.predict() returns only Infs, because
                # n_estimators within Regr is changed in second variable if
                # Regr is not newly created each iteration. debug 3 days uff
                Regr = copy.copy(Regr_orig)

                # divide into predictor and predictands
                if verbose >= 2:
                    logging.info(
                        f"{logtrunc} {varname}", "divide into predictor and predictands"
                    )
                y = data.loc[:, varname]
                notyvars = data.coords["variable"].values.tolist()
                notyvars.remove(varname)
                X = data.loc[:, notyvars]
                y_mask = mask.loc[:, varname]

                # fit dimension_reduction
                # if self.unsup is not None:
                #    X = self.unsup.fit_transform(X)

                # divide into missing and not missing values
                if verbose >= 2:
                    logging.info(f"{logtrunc} divide into missing", "and not missing")
                y_mis = y[y_mask]
                y_obs = y[~y_mask]
                del y
                # if self.unsup is not None: # TODO more elegant
                #    X_mis = datasvd[y_mask,:]
                #    X_obs = datasvd[~y_mask,:]
                # else:
                X_mis = X[y_mask.data, :]  # fall back on numpy bec variablenams
                X_obs = X[~y_mask.data, :]
                del X

                # check if missing values present in subset
                if y_mis.size == 0:
                    if verbose >= 0:
                        logging.info(
                            f"{logtrunc} variable {varname}",
                            "does not have missing values. skip ...",
                        )
                    continue

                # check if enough observations (>2) for fitting present
                if y_obs.size < 2:
                    logging.info(
                        f"{logtrunc} WARNING variable {varname}",
                        "does not have (enough) observed values. skip ...",
                    )
                    continue

                # fit regression
                if verbose >= 2:
                    logging.info(f"{logtrunc} fit to avail obs")
                if kwargs is not None:
                    Regr.fit(X_obs, y_obs, **kwargs)
                else:
                    Regr.fit(X_obs, y_obs)
                self.fittedregr[varname] = Regr

                # predict
                if verbose >= 2:
                    logging.info(f"{logtrunc} predict missing values")
                try:
                    y_predict = Regr.predict(X_mis)
                except RuntimeWarning as e:
                    raise ValueError(f"{logtrunc} {len(Regr.estimators_)}")

                # update estimates for missing values
                if verbose >= 2:
                    logging.info(f"{logtrunc} update estimates", "for missing values")
                v = np.where(data.coords["variable"] == varname)[0][0]
                # print(y_predict)
                # print(data[y_mask,v])
                data[y_mask, v] = y_predict.squeeze()
                del y_predict  # important! keep

            # calculate stopping criterion
            # troyanskaya svdimpute: change is below 0.01
            # stekhoven & bühlmann missforest: squared norm difference
            # of imputed values
            # increases for the first time ((mod-obs)**2 / mod**2)
            # self-made convergence heuristic: the denominator is default
            # the standard deviation (i think)
            # for standardised data
            # i.e. algorithm converges if change between steps is smaller
            # than epsilon times std
            self.delta_n_old = self.delta_n
            self.delta_n = (
                np.sum(np.abs(data.loc[:, varnames] - data_old.loc[:, varnames]))
                / np.mean(np.abs(data.loc[:, varnames]))
            ).values.item()

            if verbose >= 1:
                self._logiter(logtrunc)

            self.iter += 1

            # check convergence criteria
            if self.iter > self.maxiter:  # reached maximum number of iterations

                if verbose >= -1:
                    self._logiter(f"{logtrunc} TRUNCATED JOB")
                return data_old, self.fittedregr

            elif (
                np.abs(self.delta_n_old - self.delta_n) < self.epsilon
                and self.iter > self.miniter
            ):  # convergence achieved
                self.iter_below += 1

                if self.iter_below > self.miniter_below:
                    # convergence achieved already miniter_below times
                    if verbose >= -1:
                        self._logiter(f"{logtrunc} FINISHED JOB")
                    return data_old, self.fittedregr

            else:
                self.iter_below = 0
