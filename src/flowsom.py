import math
import random

import networkx as nx
from sklearn.base import BaseEstimator
import numpy as np
import pandas
import readfcs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph
from minisom import MiniSom
import igraph as ig


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
        if seed is not None:
            random.seed(seed)
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

    def build_som(self):
        # bepaal radius
        nhbrdist = squareform(pdist([(x, y) for x in range(self.xdim) for y in range(self.ydim)],metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        # som = MiniSom(
        #     x=self.xdim, y=self.ydim, input_len=num_labels, sigma=radius, learning_rate=0.05
        # )
        self.som = MiniSom(x=self.xdim, y=self.ydim, input_len=len(self.colsToUse),
                      learning_rate=0.05)
        self.som.train(self.npy_data, 100)
        print(self.som.get_weights())
        print(f'quantization error: {self.som.quantization_error(self.npy_data)}')
        self.som_weights = np.reshape(self.som.get_weights(), (self.xdim*self.ydim, len(self.colsToUse)))
        plot_SOM(self.som.get_weights(), self.xdim, self.ydim)

    def build_mst(self, networkx=True):
        if networkx:
            self.__build_mst_networkx()
        else:
            self.__build_mst_igraph()

    def __build_mst_networkx(self, clusters=None):
        nodes = self.xdim * self.ydim
        print(nodes)
        graph = nx.Graph()

        # print(som)
        print(self.som_weights.shape)
        for x in range(nodes):
            # graph.add_node()
            for y in range(x + 1, nodes):
                difference = abs(self.som_weights[x] - self.som_weights[y])
                weight = np.product(
                    [i for i in difference if not math.isnan(i)])
                graph.add_edge(x, y, weight=weight)
        tree = nx.minimum_spanning_tree(graph)
        plot_MST_networkx(tree, self.som_weights, clusters)

    def __build_mst_igraph(self, clusters=None):
        dim = self.xdim * self.ydim
        graph = ig.Graph(n=dim)
        weights = []

        for x in range(dim):
            for y in range(x + 1, dim):
                difference = abs(self.som_weights[x] - self.som_weights[y])
                weight = np.product(
                    [i for i in difference if not math.isnan(i)])
                graph.add_edges([(x, y)])
                weights.append(weight)
        graph.es["weight"] = weights
        tree = graph.spanning_tree(weights=graph.es["weight"], return_tree=True)
        plot_MST_igraph(tree, self.som_weights, clusters)

    def cluster(self, networkx=True):
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="average")
        clustering.fit(self.som_weights)
        print(clustering.labels_)
        if networkx:
            self.__build_mst_networkx(clustering.labels_)
        else:
            self.__build_mst_igraph(clustering.labels_)


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
    cols_levine_13 = ["CD45", "CD45RA", "CD19", "CD11b", "CD4", "CD8", "CD34",
                      "CD20", "CD33", "CD123", "CD38", "CD90", "CD3"]
    flowsom = FlowSom(
        input="../Gelabelde_datasets/Levine_13dim.fcs",
        colsToUse=cols_levine_13,
        seed=10
    )
    flowsom.build_som()
    flowsom.build_mst(networkx=True)
    flowsom.build_mst(networkx=False)
    flowsom.cluster(networkx=True)
    flowsom.cluster(networkx=False)
    #MST_Builder(1,1).test()
