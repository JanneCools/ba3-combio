import networkx as nx
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig


def plot_SOM(som, xdim, ydim, labels):
    fig = plt.figure(figsize=(10, 8))
    grid = GridSpec(xdim, ydim)
    for x in range(xdim):
        for y in range(ydim):
            plt.subplot(grid[x, y], aspect=1)
            plt.pie([i if i > 0 else 0 for i in som[x][y]])
    fig.legend(labels, title="Markers", title_fontsize=15)
    plt.show()
    # plt.savefig("output/som.png")


def plot_MST_networkx(tree, som, labels, clusters=None):
    fig = plt.figure(figsize=(20, 20))

    # draw edges of tree (without nodes)
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
        node_weight = [i if i > 0 else 0 for i in som[node]]
        if clusters is not None:
            color = color_map(clusters[node])
            draw_nodes(node_weight, pos[node], ax, fig, color)
        else:
            draw_nodes(node_weight, pos[node], ax, fig)
    plt.axis("equal")
    # plot the legend
    plt.legend(labels, bbox_to_anchor=[-10, 20], loc="center",
               fontsize="20", title="Markers", title_fontsize=25)
    if clusters is not None:
        plt.legend([f'Cluster {i}' for i in range(1, np.max(clusters)+1)],
                   bbox_to_anchor=[-10, 20], loc="center",
                   fontsize="20", title="Markers", title_fontsize=25)
    plt.show()
    # save the plot
    # if clusters is None:
    #     plt.savefig("../output/mst_networkx.png")
    # else:
    #     plt.savefig("output/clusters_mst_networkx.png")


def draw_nodes(data, pos, ax, fig, color: str = None):
    piesize = 0.02
    xx, yy = ax.transData.transform(pos)  # figure coordinates
    xa, ya = fig.transFigure.inverted().transform((xx, yy))  # axes coordinates
    a = plt.axes([xa-0.01, ya-0.01, piesize, piesize])
    a.set_aspect('equal')
    if color is None:
        a.pie(data)
    else:
        a.pie(data, wedgeprops={"edgecolor": color, 'linewidth': 2})


def plot_MST_igraph(tree, som, clusters=None):
    plt.figure(figsize=(20, 20))
    layout = tree.layout_kamada_kawai()

    x = [x for (x, _) in layout]
    y = [y for (_, y) in layout]

    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    ax = plt.gca()
    ax.margins(0.01)

    # plot edges
    ig.plot(
        tree,
        target=ax,
        layout=layout,
        vertex_size=0,
    )
    # get colors for the edges of the nodes
    if clusters is not None:
        color_map = plt.cm.get_cmap('rainbow', np.max(clusters)+1)
    # plot the nodes
    for node in tree.es.indices:
        node_weight = [i if i > 0 else 0 for i in som[node]]
        if clusters is not None:
            color = color_map(clusters[node])
        else:
            color = None
        ax.pie(
            node_weight, center=layout[node], radius=0.1,
            wedgeprops={"edgecolor": color, 'linewidth': 2}
        )
    plt.axis("equal")
    # if clusters is None:
    #     plt.savefig("output/mst_igraph.png")
    # else:
    #     plt.savefig("output/clustered_mst_igraph.png")
    plt.show()


