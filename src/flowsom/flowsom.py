import random
import anndata

# from plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph
from .plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph

import networkx as nx
from sklearn.base import BaseEstimator
import numpy as np
import pandas
import readfcs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn_som.som import SOM
import igraph as ig


class FlowSOM(BaseEstimator):
    def __init__(
        self,
        # input,
        pattern=".fcs",
        colsToUse=None,
        n_clusters=10,
        maxMeta=None,
        seed=10,
        xdim=10,
        ydim=10
    ):
        self.input = input
        self.pattern = pattern
        self.maxMeta = maxMeta
        self.seed = seed
        self.xdim = xdim
        self.ydim = ydim
        self.colsToUse = colsToUse
        self.n_clusters = n_clusters
        self.adata = None
        self.som = None
        self.np_data = None
        random.seed(seed)

    def remove_unused_data(self, columns):
        cols = self.adata.uns["meta"]["channels"]["$PnN"]
        indices = cols.index.astype(np.intc)
        unused = [indices[i] for i, col in enumerate(cols) if col not in columns]
        unused = np.flip(np.sort(unused))
        data = self.adata.X
        for index in unused:
            data = np.delete(data, index-1, axis=1)
        # sla de data op van de kolommen die gebruikt moeten worden
        self.adata.uns["used_data"] = data
        print(data)

    def build_som(self, xdim, ydim, cols):
        data = self.adata.uns["used_data"]

        # bepaal radius
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        self.som = SOM(
            m=xdim, n=ydim, dim=cols, sigma=radius/2, lr=0.05,
            random_state=self.seed
        )
        # self.som = SOM(
        #     m=xdim, n=ydim, dim=cols, lr=0.05, random_state=self.seed
        # )
        self.som.fit(data)

        # update anndata
        self.adata.uns["som_weights"] = np.reshape(self.som.cluster_centers_, (xdim*ydim, cols))

        plot_SOM(self.som.cluster_centers_, xdim, ydim)

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
            for y in range(x + 1, nodes):
                weight = np.sum(abs(weights[x] - weights[y]))
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
                weight = np.sum(abs(som_weights[x] - som_weights[y]))
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
        clustering.fit(self.adata.uns["som_weights"])

        # update anndata
        self.adata.uns["cluster_labels"] = clustering.labels_
        if networkx:
            self.__build_mst_networkx(xdim, ydim, clustering.labels_)
        else:
            self.__build_mst_igraph(xdim, ydim, clustering.labels_)

    def set_params(self, **params):
        self.__dict__.update(params)

    def fit(self, x, y=None):
        if isinstance(x, str):
            self.adata = readfcs.read(x)
        elif isinstance(x, np.ndarray):
            self.adata = anndata.AnnData(x)
        elif isinstance(x, pandas.DataFrame):
            self.adata = anndata.AnnData(x)
        if self.colsToUse is None:
            self.colsToUse = self.adata.var_names
            self.adata.uns["used_data"] = self.adata.X
            # self.adata.uns["meta"]["channels"]["$PnN"] = self.colsToUse
        else:
            self.remove_unused_data(self.colsToUse)

        # build SOM
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))

        # metaclustering
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        return self

    def predict(self, x):
        if isinstance(x, str):
            adata = readfcs.read(x)
        elif isinstance(x, np.ndarray):
            adata = anndata.AnnData(x)
        else:
            adata = x
        data = adata.X
        # find som winner for every point in x
        winners = self.som.predict(data)
        clusters = [self.adata.uns["cluster_labels"][i] for i in winners]
        return clusters

    def fit_predict(self, x, y=None):
        if isinstance(x, str):
            self.adata = readfcs.read(x)
        elif isinstance(x, np.ndarray):
            self.adata = anndata.AnnData(x)
        elif isinstance(x, pandas.DataFrame):
            self.adata = anndata.AnnData(x)
        if self.colsToUse is None:
            self.colsToUse = self.adata.var_names
            self.adata.uns["used_data"] = self.adata.X
            # self.adata.uns["meta"]["channels"]["$PnN"] = self.colsToUse
        else:
            self.remove_unused_data(self.colsToUse)

        # build SOM
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))

        # metaclustering
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        # predict
        # find som winner for every point in x
        winners = self.som.predict(self.adata.uns["used_data"])
        clusters = [self.adata.uns["cluster_labels"][i] for i in winners]
        return clusters



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
        report = open("../../verslag.pdf", "r")
        lines = report.readlines()

        output = open(filename, "w")
        output.writelines(lines)

