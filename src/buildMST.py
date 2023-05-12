import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from networkx.drawing.nx_pydot import graphviz_layout


class MST_Builder():
    def __init__(self, xdim, ydim):
        self.xdim = xdim
        self.ydim = ydim

    # def build_mst_igraph(self, som: list, channels: int):
    #     nodes = self.xdim*self.ydim
    #     graph = ig.Graph()
    #     som = np.reshape(som, (nodes, channels))
    #     # print(som)
    #     print(som.shape)
    #     for x in range(nodes):
    #         for y in range(x + 1, nodes):
    #             difference = abs(som[x] - som[y])
    #             weight = np.product(
    #                 [i for i in difference if not math.isnan(i)])
    #             graph.add_edge(x, y, weight=weight)
    #     tree = graph.spanning_tree()
    #
    #     pos = graphviz_layout(tree, prog="neato")
    #     ig.drawing.plot(tree, pos=pos)
    #     ax = plt.gca()
    #     ax.margins(0.1)
    #     plt.axis("off")
    #     #plt.savefig("twopi_mst_edges")
    #     plt.show()

    def build_mst(self, som: list, channels: int):
        nodes = self.xdim*self.ydim
        print(nodes)
        graph = nx.Graph()
        #graph.add_nodes_from([i for i in range(nodes)])
        som = np.reshape(som, (nodes, channels))
        #print(som)
        print(som.shape)
        for x in range(nodes):
            for y in range(x+1, nodes):
                difference = abs(som[x]-som[y])
                weight = np.product([i for i in difference if not math.isnan(i)])
                graph.add_edge(x, y, weight=weight)
        tree = nx.minimum_spanning_tree(graph)

        a = np.array(som)

        pos = graphviz_layout(tree, prog="neato")
        nx.draw_networkx_edges(tree, pos)
        # for node in tree.nodes:
        #     node_weights = [i if not math.isnan(i) and i > 0 else 0 for i in som[node]]
        #     plt.subplot(pos[node][0], pos[node][1], aspect=1)
        #     a = plt.pie(
        #         node_weights,
        #         center=pos[node],
        #         radius=0.1
        #     )
        # tree.nodes[0] = plt.pie([i if not math.isnan(i) and i > 0 else 0 for i in som[0]])
        nx.draw_networkx_nodes(tree, pos, node_size=100)

        ax = plt.gca()
        ax.margins(0.2)
        plt.axis("off")
        #plt.savefig("twopi_mst_edges")
        plt.show()