import math

import networkx as nx
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import igraph as ig


def plot_SOM(som, xdim, ydim):
    grid = GridSpec(xdim, ydim)
    for x in range(xdim):
        for y in range(ydim):
            plt.subplot(grid[x, y], aspect=1)
            plt.pie([i if i > 0 else 0 for i in som[x][y]])
    # self.som = som
    # plt.savefig("output2.png")
    plt.show()


def plot_MST_networkx(tree, som, clusters=None):
    plt.figure(figsize=(20, 20))
    pos = graphviz_layout(tree, prog="neato")
    nx.draw_networkx_edges(tree, pos)
    # nx.draw_networkx_nodes(tree, pos, node_size=100)

    ax = plt.gca()
    ax.margins(0.01)

    # get colors for the edges of the nodes
    if clusters is not None:
        color_map = plt.cm.get_cmap('hsv', np.max(clusters)+1)
    # plot the nodes
    for node in tree.nodes:
        node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in
                       som[node]]
        node_pos = pos[node]
        if clusters is not None:
            color = color_map(clusters[node])
        else:
            color = None
        draw_nodes(node_weight, node_pos[0], node_pos[1], ax, color)
    plt.axis("off")
    # plt.savefig("twopi_mst_edges")
    plt.show()


def plot_MST_igraph(tree, som, clusters=None):
    plt.figure(figsize=(20, 20))
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
    # get colors for the edges of the nodes
    if clusters is not None:
        color_map = plt.cm.get_cmap('hsv', np.max(clusters)+1)
    # plot the nodes
    for node in tree.es.indices:
        node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in
                       som[node]]
        node_pos = layout[node]
        if clusters is not None:
            color = color_map(clusters[node])
        else:
            color = None
        draw_nodes(node_weight, node_pos[0], node_pos[1], ax, color)
    # plt.pie([0.5,0.5], center=(0,0), radius=0.1)
    plt.axis("off")
    plt.show()

# https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
def draw_nodes(data, xpos, ypos, ax, color):

    # for incremental pie slices
    cumsum = np.cumsum(data)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()

    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()
        xy = np.column_stack([x, y])
        if color is not None:
            ax.scatter([xpos], [ypos], marker=xy, s=1800, edgecolors=color, linewidth=3)
        else:
            ax.scatter([xpos], [ypos], marker=xy, s=1800)
    return ax
