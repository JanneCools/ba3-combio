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
    plt.savefig("output/som.png")
    plt.show()


def plot_MST_networkx(tree, som, clusters=None):
    fig = plt.figure(figsize=(20, 20))
    # pos = graphviz_layout(tree, prog="neato")
    pos = nx.nx_agraph.graphviz_layout(tree, prog="neato")
    nx.draw_networkx_edges(tree, pos, width=3)

    x = [x for (x, _) in pos.values()]
    y = [y for (_, y) in pos.values()]

    plt.xlim(np.min(x)-10, np.max(x)+10)
    plt.ylim(np.min(y)-10, np.max(y)+10)

    ax = plt.gca()
    ax.margins(0.01)

    # get colors for the edges of the nodes
    if clusters is not None:
        color_map = plt.cm.get_cmap('rainbow', np.max(clusters)+1)
    # plot the nodes
    for node in tree.nodes:
        node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in
                       som[node]]
        if clusters is not None:
            color = color_map(clusters[node])
        else:
            color = None
        draw_nodes(node_weight, pos[node], ax, fig, color)
    plt.axis("off")
    if clusters is None:
        plt.savefig("output/mst_networkx.png")
    else:
        plt.savefig("output/clustered_mst_networkx.png")
    plt.show()


def plot_MST_igraph(tree, som, clusters=None):
    fig = plt.figure(figsize=(20, 20))
    layout = tree.layout_kamada_kawai()
    # fig, ax = plt.subplots()

    # x = [x for (x, _) in layout.values()]
    # y = [y for (_, y) in layout.values()]
    #
    # plt.xlim(np.min(x)-10, np.max(x)+10)
    # plt.ylim(np.min(y)-10, np.max(y)+10)

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
        color_map = plt.cm.get_cmap('rainbow', np.max(clusters)+1)
    # plot the nodes
    for node in tree.es.indices:
        node_weight = [i if not math.isnan(i) and i > 0 else 0 for i in
                       som[node]]
        if clusters is not None:
            color = color_map(clusters[node])
        else:
            color = None
        draw_nodes(node_weight, layout[node], ax, fig, color)
    # plt.pie([0.5,0.5], center=(0,0), radius=0.1)
    plt.axis("off")
    if clusters is None:
        plt.savefig("output/mst_igraph.png")
    else:
        plt.savefig("output/clustered_mst_igraph.png")
    plt.show()


def draw_nodes(data, pos, ax, fig, color):
    piesize = 0.03
    xx, yy = ax.transData.transform(pos)  # figure coordinates
    xa, ya = fig.transFigure.inverted().transform((xx, yy))  # axes coordinates
    a = plt.axes([xa-0.015, ya-0.015, piesize, piesize])
    a.set_aspect('equal')
    if color is None:
        a.pie(data)
    else:
        a.pie(data, wedgeprops={"edgecolor": color, 'linewidth': 2})

