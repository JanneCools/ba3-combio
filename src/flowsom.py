from sklearn.base import BaseEstimator
import numpy as np
import pandas
from flowio import FlowData
from buildSOM import SOM_Builder


class FlowSom(BaseEstimator):
    def __init__(
        self,
        input,
        pattern=".fcs",
        silent=True,
        colsToUse=None,
        n_clusters=10,
        maxMeta=None,
        importance=None,
        seed=None,
    ):
        self.input = input
        self.pattern = pattern
        self.silent = silent
        self.colsToUse = colsToUse
        self.n_clusters = n_clusters
        self.maxMeta = maxMeta
        self.importance = importance
        self.seed = seed
        self.som = None
        if isinstance(input, str):
            self.fcs_data = FlowData(self.input)
            self.npy_data = np.reshape(self.fcs_data.events, (-1, self.fcs_data.channel_count))


    def buildSOM(self):
        som_builder = SOM_Builder(self.n_clusters, self.n_clusters)
        self.som = som_builder.som(self.npy_data, self.fcs_data.channel_count)

    def set_params(self, **params):
        self.__dict__.update(params)

    def fit(self, X: pandas.DataFrame, y=None):
        raise NotImplementedError

    def predict(self, X: pandas.DataFrame):
        raise NotImplementedError

    def fit_predict(self, X: pandas.DataFrame, y=None):
        raise NotImplementedError

    def as_df(self, lazy=True):
        """
        :param lazy: dask if lazy else pandas
        :return: DataFrame van de ingegeven data
        """
        raise NotImplementedError

    def as_adata(self, lazy=True):
        """
        :param lazy: dask if lazy else pandas
        :return: AnnData met in .uns de clustering in een FlowSom
        """


if __name__ == "__main__":
    flowsom = FlowSom(input="../Gelabelde_datasets/FlowCAP_WNV.fcs")
    flowsom.buildSOM()
