from .plotting import plot_SOM, plot_MST_networkx, plot_MST_igraph
from .util import SOM_builder, read_input
from .report import write_intro, write_som, write_mst, write_metaclustering

import networkx as nx
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import random

from fpdf import FPDF
import time


class FlowSOM(BaseEstimator):
    def __init__(
        self,
        colsToUse=None,
        n_clusters=8,
        silent=False,
        seed=10,
        xdim=10,
        ydim=10,
        networkx=True,
        igraph=False,
        minisom=False,
    ):
        # parameters for algorithms
        self.xdim = xdim
        self.ydim = ydim
        self.colsToUse = colsToUse
        self.n_clusters = n_clusters
        self.silent = silent
        self.networkx = networkx
        self.igraph = igraph
        self.minisom = minisom
        self.seed = seed
        random.seed(self.seed)

        # data to be stored
        self.adata = None
        self.som = None
        self.np_data = None

        # data for the report
        self.pdf = FPDF()
        self.pdf.add_page()
        self.som_time = 0
        self.clustering_time = 0

    def build_som(self, xdim: int, ydim: int, cols: int):
        """
        Build the self-organising map
        :param xdim: the x dimension of the grid
        :param ydim: the y dimension of the grid
        :param cols: the amount of columns (markers) that are used
        :return: None
        """
        data = self.adata.uns["used_data"]

        start = time.process_time()
        # calculate radius
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        # make SOM
        self.som = SOM_builder(
            xdim=xdim,
            ydim=ydim,
            cols=cols,
            radius=radius / 3,
            alpha=0.05,
            seed=self.seed,
            minisom=self.minisom,
        )
        clusters = self.som.fit(data)
        stop = time.process_time()
        self.som_time = stop - start
        if not self.silent:
            print(f"start: {start}\nstop: {stop}\nTijd voor SOM: {stop-start}")

        # update anndata
        temp = np.reshape(clusters, (xdim * ydim, cols))
        self.adata.uns["som_clusters"] = np.nan_to_num(temp)

        # plot SOM
        plot_SOM(clusters, xdim, ydim, self.colsToUse)
        self.build_mst(xdim, ydim)

    def build_mst(self, xdim: int, ydim: int):
        """
        Build a minimal spanning tree of the SOM
        :param xdim: the x dimension of the grid
        :param ydim: the y dimension of the grid
        :return: None
        """
        if self.networkx:
            self.__build_mst_networkx(xdim, ydim)
        if self.igraph:
            self.__build_mst_igraph(xdim, ydim)

    def __build_mst_networkx(self, xdim: int, ydim: int, clusters=None):
        """
        Build the MST using NetworkX
        :param xdim: the x dimension of the grid
        :param ydim: the y dimension of the grid
        :param clusters: the meta-clusters that each SOM-node correponds to
        :return: None
        """
        # build a graph where each node is connected to each node
        # the weight of the edge is calculated based on both their SOM-weights
        nodes = xdim * ydim
        graph = nx.Graph()
        weights = self.adata.uns["som_clusters"]
        for x in range(nodes):
            for y in range(x + 1, nodes):
                weight = np.sum(abs(weights[x] - weights[y]))
                graph.add_edge(x, y, weight=weight)

        # turn the graph into a minimal spanning tree
        tree = nx.minimum_spanning_tree(graph)

        if clusters is not None:
            self.adata.uns["mst_clustering"] = tree
        else:
            self.adata.uns["mst_som"] = tree
        plot_MST_networkx(tree, weights, self.colsToUse, clusters)

    def __build_mst_igraph(self, xdim: int, ydim: int, clusters=None):
        """
        Build the MST using IGraph
        :param xdim: the x dimension of the grid
        :param ydim: the y dimension of the grid
        :param clusters: the meta-clusters that each SOM-node correponds to
        :return: None
        """
        # build a graph where each node is connected to each node
        # the weight of the edge is calculated based on both their SOM-weights
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

        # turn the graph into a minimal spanning tree
        tree = graph.spanning_tree(weights=graph.es["weight"], return_tree=True)
        if clusters is not None:
            self.adata.uns["mst_clustering"] = tree
        else:
            self.adata.uns["mst_som"] = tree
        plot_MST_igraph(tree, som_weights, self.colsToUse, clusters)

    def cluster(self, n_clusters: int, xdim: int, ydim: int):
        """
        Generate the meta-clusters
        :param n_clusters: the amount of meta-clusters to be generated
        :param xdim: the x dimension of the grid
        :param ydim: the y dimension of the grid
        :return: None
        """
        start = time.process_time()
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
        clustering.fit(self.adata.uns["som_clusters"])
        stop = time.process_time()
        self.clustering_time = stop - start
        if not self.silent:
            print(
                f"Start: {start}\nStop: {stop}\nTijd voor metaclustering: {stop-start}"
            )

        # update anndata
        self.adata.uns["metaclusters"] = clustering.labels_
        if self.networkx:
            self.__build_mst_networkx(xdim, ydim, clustering.labels_)
        if self.igraph:
            self.__build_mst_igraph(xdim, ydim, clustering.labels_)

    def set_params(self, **params):
        self.__dict__.update(params)

    def fit(self, x, dataset_name=None, y=None):
        self.adata = read_input(x, self.colsToUse, self.adata)
        if self.colsToUse is None or "meta" not in self.adata.uns:
            self.colsToUse = self.adata.var_names

        # write intro in report
        write_intro(
            pdf=self.pdf,
            size=len(self.adata.X),
            colsToUse=self.colsToUse,
            xdim=self.xdim,
            ydim=self.ydim,
            dataset_name=dataset_name,
        )

        # build SOM and perform meta-clustering
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        # add SOM, mst and metaclustering to report
        write_som(pdf=self.pdf, minisom=self.minisom, time=self.som_time)
        write_mst(pdf=self.pdf, networkx=self.networkx, igraph=self.igraph)
        write_metaclustering(
            pdf=self.pdf,
            n_clusters=self.n_clusters,
            networkx=self.networkx,
            igraph=self.igraph,
            time=self.clustering_time,
        )
        return self

    def predict(self, x):
        adata = read_input(x, self.colsToUse)
        data = adata.uns["used_data"]

        # find util winner for every point in x
        winners = self.som.predict(data)
        clusters = [self.adata.uns["metaclusters"][i] for i in winners]
        return clusters

    def fit_predict(self, x, dataset_name=None, y=None):
        self.adata = read_input(x, self.colsToUse, self.adata)
        if self.colsToUse is None or "meta" not in self.adata.uns:
            self.colsToUse = self.adata.var_names
        # write intro in report
        write_intro(
            pdf=self.pdf,
            size=len(self.adata.X),
            colsToUse=self.colsToUse,
            xdim=self.xdim,
            ydim=self.ydim,
            dataset_name=dataset_name,
        )

        # build SOM
        self.build_som(self.xdim, self.ydim, len(self.colsToUse))
        # metaclustering
        self.cluster(self.n_clusters, self.xdim, self.ydim)

        # add SOM, mst and metaclustering to report
        write_som(pdf=self.pdf, minisom=self.minisom, time=self.som_time)
        write_mst(pdf=self.pdf, networkx=self.networkx, igraph=self.igraph)
        write_metaclustering(
            pdf=self.pdf,
            n_clusters=self.n_clusters,
            networkx=self.networkx,
            igraph=self.igraph,
            time=self.clustering_time,
        )

        # find metacluster for every point in x
        winners = self.som.predict(self.adata.uns["used_data"])
        clusters = [self.adata.uns["metaclusters"][i] for i in winners]
        return clusters

    def as_df(self, lazy=True, copy=False):
        """
        :param lazy: dask if lazy else pandas
        :param copy: whether to copy the data or not
        :return: DataFrame of the data
        """
        adata = self.as_adata(lazy=lazy, copy=copy)
        if lazy:
            import dask

            return dask.DataFrame.from_dask_array(
                adata.X, columns=adata.var_names, index=adata.obs_names
            )
        return pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)

    def as_adata(self, lazy=True, copy=False):
        """
        :param copy: whether to copy or not
        :param lazy: dask if lazy else pandas
        :return: AnnData object
        """
        adata = self.adata
        if lazy:
            # only works if AnnData was created lazily
            return adata
        return adata.to_memory(copy=copy)

    def report(self, filename: str):
        """
        Write the report to the filename
        :param filename: file to which the report should be written
        :return: None
        """
        self.pdf.output(filename)
