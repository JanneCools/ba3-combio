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

    def build_mst_igraph(self, som: list, channels: int):
        nodes = self.xdim*self.ydim
        graph = ig.Graph(n=nodes)
        weights = []
        som = np.reshape(som, (nodes, channels))

        for x in range(nodes):
            for y in range(x + 1, nodes):
                difference = abs(som[x] - som[y])
                weight = np.product(
                    [i for i in difference if not math.isnan(i)])
                graph.add_edges([(x, y)])
                weights.append(weight)
        graph.es["weight"] = weights
        tree = graph.spanning_tree(weights=graph.es["weight"], return_tree=True)

        # pos = graphviz_layout(tree, prog="neato")
        layout = tree.layout_kamada_kawai()
        # fig, ax = plt.subplots()
        ax = plt.gca()
        ax.margins(0.01)

        # plot edges
        ig.plot(
            tree,
            target=ax,
            layout=layout,
            vertex_size=0
        )
        for node in tree.es.indices:
            node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in som[node]]
            node_pos = layout[node]
            self.draw_nodes(node_weight, node_pos[0], node_pos[1], ax)
        # plt.pie([0.5,0.5], center=(0,0), radius=0.1)
        plt.axis("off")
        plt.show()

    def build_mst(self, som: list, channels: int):
        nodes = self.xdim*self.ydim
        print(nodes)
        graph = nx.Graph()

        som = np.reshape(som, (nodes, channels))
        #print(som)
        print(som.shape)
        for x in range(nodes):
            # graph.add_node()
            for y in range(x+1, nodes):
                difference = abs(som[x]-som[y])
                weight = np.product([i for i in difference if not math.isnan(i)])
                graph.add_edge(x, y, weight=weight)
        tree = nx.minimum_spanning_tree(graph)

        pos = graphviz_layout(tree, prog="neato")
        nx.draw_networkx_edges(tree, pos)
        # nx.draw_networkx_nodes(tree, pos, node_size=100)

        ax = plt.gca()
        ax.margins(0.01)

        for node in tree.nodes:
            node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in som[node]]
            node_pos = pos[node]
            self.draw_nodes(node_weight, node_pos[0], node_pos[1], ax)

        plt.axis("off")
        #plt.savefig("twopi_mst_edges")
        plt.show()

    # https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    @staticmethod
    def draw_nodes(data, xpos, ypos, ax):
        # for incremental pie slices
        cumsum = np.cumsum(data)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        for r1, r2 in zip(pie[:-1], pie[1:]):
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            ax.scatter([xpos], [ypos], marker=xy, s=100)

        return ax
