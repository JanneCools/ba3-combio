import math
import random

from .plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph

import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas
import readfcs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from minisom import MiniSom
import igraph as ig


class FlowSOM(BaseEstimator):
    def __init__(
        self,
        input,
        pattern=".fcs",
        silent=True,
        colsToUse=None,
        n_clusters=10,
        maxMeta=None,
        seed=None,
        xdim=10,
        ydim=10
    ):
        self.input = input
        self.pattern = pattern
        self.silent = silent
        self.maxMeta = maxMeta
        self.seed = seed
        self.xdim = xdim
        self.ydim = ydim
        self.cols = len(colsToUse)
        self.som = None
        self.np_data = None
        if seed is not None:
            random.seed(seed)
        if isinstance(input, str):
            self.adata = readfcs.read(self.input)
            self.remove_unused_data(colsToUse)
        self.build_som(xdim, ydim, len(colsToUse))
        self.build_mst(xdim, ydim, networkx=True)
        self.cluster(n_clusters, xdim, ydim, networkx=True)

    def remove_unused_data(self, columns):
        cols = self.adata.uns["meta"]["channels"]["$PnN"]
        indices = cols.index.astype(np.intc)
        unused = [indices[i] for i, col in enumerate(cols) if col not in columns]
        unused = np.flip(np.sort(unused))
        data = self.adata.X
        for index in unused:
            data = np.delete(data, index-1, axis=1)
        self.np_data = np.nan_to_num(data)
        # sla de data op van de kolommen die gebruikt moeten worden
        self.adata.uns["used_data"] = self.np_data

    def build_som(self, xdim, ydim, cols):
        # bepaal radius
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        # self.som = MiniSom(
        #     x=xdim, y=ydim, input_len=self.np_data.shape[1],
        #     sigma=radius, learning_rate=0.05, random_seed=self.seed
        # )
        self.som = MiniSom(
            x=xdim, y=ydim, input_len=cols, learning_rate=0.05, random_seed=self.seed
        )
        self.som.train(self.np_data, 100, verbose=True)
        self.adata.uns["som_weights"] = np.reshape(self.som.get_weights(), (xdim*ydim, cols))
        # win = self.som.win_map(self.np_data)
        # print(win)

        # update anndata
        self.adata.uns["som_weights"] = self.adata.uns["som_weights"]

        plot_SOM(self.som.get_weights(), xdim, ydim)

    def build_mst(self, xdim, ydim, networkx=True):
        if networkx:
            self.__build_mst_networkx(xdim, ydim)
        else:
            self.__build_mst_igraph(xdim, ydim)

    def __build_mst_networkx(self, xdim, ydim, clusters=None):
        nodes = xdim * ydim
        print(nodes)
        graph = nx.Graph()

        # print(som)
        weights = self.adata.uns["som_weights"]
        print(weights.shape)
        for x in range(nodes):
            # graph.add_node()
            for y in range(x + 1, nodes):
                difference = abs(weights[x] - weights[y])
                weight = np.sum(
                         [i for i in difference if not math.isnan(i)])
                graph.add_edge(x, y, weight=weight)
        tree = nx.minimum_spanning_tree(graph)
        if clusters is not None:
            self.adata.uns["mst_clustering"] = tree
        else:
            self.adata.uns["mst_som"] = tree
        plot_MST_networkx(tree, weights, clusters)

    def __build_mst_igraph(self, xdim, ydim, clusters=None):
        dim = xdim * ydim
        graph = ig.Graph(n=dim)
        weights = []

        som_weights = self.adata.uns["som_weights"]
        for x in range(dim):
            for y in range(x + 1, dim):
                difference = abs(som_weights[x] - som_weights[y])
                weight = np.sum(
                    [i for i in difference if not math.isnan(i)])
                graph.add_edges([(x, y)])
                weights.append(weight)
        graph.es["weight"] = weights
        tree = graph.spanning_tree(weights=graph.es["weight"], return_tree=True)
        if clusters is not None:
            self.adata.uns["mst_clustering"] = tree
        else:
            self.adata.uns["mst_som"] = tree
        plot_MST_igraph(tree, som_weights, clusters)

    def cluster(self, n_clusters, xdim, ydim, networkx=True):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
        som_weights = np.nan_to_num(self.adata.uns["som_weights"])
        clustering.fit(som_weights)
        print(clustering.labels_)
        self.adata.uns["cluster_labels"] = clustering.labels_
        if networkx:
            self.__build_mst_networkx(xdim, ydim, clustering.labels_)
        else:
            self.__build_mst_igraph(xdim, ydim, clustering.labels_)


    def set_params(self, **params):
        self.__dict__.update(params)

    def fit(self, X: pandas.DataFrame, y=None):
        x = ClassifierMixin()
        # x.
        # self.build_som(self.xdim, self.ydim, self.cols)
        # print(X)
        # estimator = BaseEstimator()
        # estimator.fit(X.values, self.som_weights)
        return self
        # raise NotImplementedError

    def predict(self, X: pandas.DataFrame):
        raise NotImplementedError

    def fit_predict(self, X: pandas.DataFrame, y=None):
        raise NotImplementedError

    def as_df(self, lazy=True):
        """
        :param lazy: dask if lazy else pandas
        :return: DataFrame van de ingegeven data
        """
        # raise NotImplementedError
        return self.adata.to_df()

    def as_adata(self, lazy=True):
        """
        :param lazy: dask if lazy else pandas
        :return: AnnData met in .uns de clustering in een FlowSom
        """
        return self.adata

    def report(self, filename: str):
        report = open("../verslag.pdf", "r")
        lines = report.readlines()

        output = open(filename, "w")
        output.writelines(lines)


if __name__ == "__main__":
    cols_flowcap_nd = ["FITC-A", "PerCP-Cy5-5-A", "Pacific Blue-A",
                       "Pacifc Orange-A", "QDot 605-A", "APC-A", "Alexa 700-A",
                       "PE-A", "PE-Cy5-A", "PE-Cy7-A"]
    cols_levine_13 = ["CD45", "CD45RA", "CD19", "CD11b", "CD4", "CD8", "CD34",
                      "CD20", "CD33", "CD123", "CD38", "CD90", "CD3"]
    cols_levine_32 = ["CD45RA", "CD133", "CD19", "CD22", "CD11b", "CD4", "CD8",
                      "CD34", "Flt3", "CD20", "CXCR4", "CD235ab", "CD45",
                      "CD123", "CD321", "CD14", "CD33", "CD47", "CD11c", "CD7",
                      "CD15", "CD16", "CD44", "CD38", "CD13", "CD3", "CD61",
                      "CD117", "CD49d", "HLA-DR", "CD64", "CD41"]
    flowsom = FlowSOM(
        input="../../Gelabelde_datasets/FlowCAP_ND.fcs",
        colsToUse=cols_flowcap_nd,
        seed=10
    )
    # adata = flowsom.as_adata()
    # used_data = adata.uns["used_data"]
    # flowsom.fit(pandas.DataFrame(used_data))
    # flowsom.report("test.txt")
