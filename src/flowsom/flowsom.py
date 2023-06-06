import random
import anndata
import fpdf

from .plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph
from .som import SOM_builder

import networkx as nx
from sklearn.base import BaseEstimator
import numpy as np
import pandas
import readfcs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import igraph as ig

from PIL import Image
from fpdf import FPDF

class FlowSOM(BaseEstimator):
    def __init__(
        self,
        # input,
        pattern=".fcs",
        colsToUse=None,
        n_clusters=8,
        maxMeta=None,
        seed=10,
        xdim=10,
        ydim=10,
        networkx=True,
        igraph=False,
        minisom=False
    ):
        self.input = input
        self.pattern = pattern
        self.maxMeta = maxMeta
        self.seed = seed
        self.xdim = xdim
        self.ydim = ydim
        self.colsToUse = colsToUse
        self.n_clusters = n_clusters
        self.networkx = networkx
        self.igraph = igraph
        self.minisom = minisom
        self.adata = None
        self.som = None
        self.np_data = None
        self.pdf = FPDF()
        self.pdf.add_page()
        random.seed(seed)

    def read_input(self, inp):
        # make anndata object
        if isinstance(inp, str):
            self.adata = readfcs.read(inp)
        elif isinstance(inp, np.ndarray) or isinstance(inp, pandas.DataFrame):
            self.adata = anndata.AnnData(inp)

        # remove unused columns
        if self.colsToUse is None or "meta" not in self.adata.uns:
            self.colsToUse = self.adata.var_names
            self.adata.uns["used_data"] = self.adata.X
        else:
            self.remove_unused_data(self.colsToUse)

    def remove_unused_data(self, columns):
        cols = self.adata.uns["meta"]["channels"]["$PnN"]
        indices = cols.index.astype(np.intc)
        unused = [indices[i] for i, col in enumerate(cols) if col not in columns]
        unused = np.flip(np.sort(unused))
        data = self.adata.X
        for index in unused:
            data = np.delete(data, index-1, axis=1)
        # only save the data from the used columns
        self.adata.uns["used_data"] = data

    def build_som(self, xdim, ydim, cols):
        data = self.adata.uns["used_data"]

        # calculate radius
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        # make SOM
        # self.som = SOM(
        #     m=xdim, n=ydim, dim=cols, sigma=radius/3, lr=0.05,
        #     random_state=self.seed
        # )
        self.som = SOM_builder(
            xdim=xdim, ydim=ydim, cols=cols, radius=radius/3, alpha=0.05,
            seed=self.seed, minisom=self.minisom
        )
        clusters = self.som.fit(data)

        # update anndata
        self.adata.uns["som_clusters"] = np.reshape(clusters, (xdim*ydim, cols))

        # plot SOM
        plot_SOM(clusters, xdim, ydim, self.colsToUse)
        self.build_mst(xdim, ydim)

    def build_mst(self, xdim, ydim):
        if self.networkx:
            self.__build_mst_networkx(xdim, ydim)
        if self.igraph:
            self.__build_mst_igraph(xdim, ydim)

    def __build_mst_networkx(self, xdim, ydim, clusters=None):
        nodes = xdim * ydim
        graph = nx.Graph()

        # print(som)
        weights = self.adata.uns["som_clusters"]
        for x in range(nodes):
            for y in range(x + 1, nodes):
                weight = np.sum(abs(weights[x] - weights[y]))
                graph.add_edge(x, y, weight=weight)
        tree = nx.minimum_spanning_tree(graph)
        if clusters is not None:
            self.adata.uns["mst_clustering"] = tree
        else:
            self.adata.uns["mst_som"] = tree
        plot_MST_networkx(tree, weights, self.colsToUse, clusters)

    def __build_mst_igraph(self, xdim, ydim, clusters=None):
        dim = xdim * ydim
        graph = ig.Graph(n=dim)
        weights = []

        som_weights = self.adata.uns["som_clusters"]
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

    def cluster(self, n_clusters, xdim, ydim):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="average"
        )
        clustering.fit(self.adata.uns["som_clusters"])

        # update anndata
        self.adata.uns["metaclusters"] = clustering.labels_
        if self.networkx:
            self.__build_mst_networkx(xdim, ydim, clustering.labels_)
        if self.igraph:
            self.__build_mst_igraph(xdim, ydim, clustering.labels_)

    def set_params(self, **params):
        self.__dict__.update(params)

    def fit(self, x, dataset_name=None, y=None):
        self.read_input(x)
        # write report
        self.pdf.set_font('Arial', '', 18)
        if dataset_name is None:
            self.pdf.write(5, "FlowSOM algoritme\n")
        else:
            self.pdf.write(
                10,
                f"FlowSOM algoritme toegepast op de dataset {dataset_name}\n\n"
            )
        # build SOM
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))
        # perform meta-clustering
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        # write report
        self.pdf.set_font('Arial', '', 14)
        self.pdf.cell(w=100, h=100, txt="Self orginising map of the dataset")
        # self.pdf.write(5, "Self organising map of the dataset:")
        self.pdf.image("som.jpg", w=80, h=60)
        self.pdf.write(5, "Minimal spanning tree of the SOM:\n")
        self.pdf.image("mst_networkx.jpg", w=80, h=80)
        self.pdf.write(5, "Metaclusters of the SOM:\n")
        self.pdf.image("clusters_mst_networkx.jpg", w=100, h=100)
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
        clusters = [self.adata.uns["metaclusters"][i] for i in winners]
        return clusters

    def fit_predict(self, x, y=None):
        self.read_input(x)
        # build SOM
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))
        # metaclustering
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        # find metacluster for every point in x
        winners = self.som.predict(self.adata.uns["used_data"])
        clusters = [self.adata.uns["metaclusters"][i] for i in winners]
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
        self.pdf.output(filename)


