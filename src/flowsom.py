from sklearn.base import BaseEstimator
import numpy as np
import pandas
import anndata
from flowio import FlowData
import readfcs
from buildSOM import SOM_Builder
from buildMST import MST_Builder


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
        xdim=10,
        ydim=10
    ):
        self.input = input
        self.pattern = pattern
        self.silent = silent
        self.colsToUse = colsToUse
        self.n_clusters = n_clusters
        self.maxMeta = maxMeta
        self.importance = importance
        self.seed = seed
        self.xdim = xdim
        self.ydim = ydim
        self.som = None
        self.npy_data = None
        if isinstance(input, str):
            self.adata = readfcs.read(self.input)
            self.remove_unused_data()

    def remove_unused_data(self):
        cols = self.adata.var_names
        indices = [i for i, col in enumerate(cols) if col not in self.colsToUse]
        indices = np.flip(np.sort(indices))
        data = self.adata.X
        for index in indices:
            data = np.delete(data, index, axis=1)
        self.npy_data = data

    def buildSOM(self):
        som_builder = SOM_Builder(self.xdim, self.ydim)
        self.som = som_builder.build(self.npy_data, len(self.colsToUse))

    def buildMST(self):
        mst_builder = MST_Builder(self.xdim, self.ydim)
        # mst_builder.build_mst_igraph(self.som, len(self.colsToUse))
        mst_builder.build_mst(self.som, len(self.colsToUse))

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
    cols_flowcap_nd = ["FITC-A", "PerCP-Cy5-5-A", "Pacific Blue-A",
                       "Pacifc Orange-A", "QDot 605-A", "APC-A", "Alexa 700-A",
                       "PE-A", "PE-Cy5-A", "PE-Cy7-A"]
    flowsom = FlowSom(
        input="../Gelabelde_datasets/FlowCAP_ND.fcs",
        colsToUse=cols_flowcap_nd
    )
    flowsom.buildSOM()
    flowsom.buildMST()
    #MST_Builder(1,1).test()
