import networkx as nx
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import igraph as ig


def plot_SOM(som, xdim, ydim, labels):
    # prepare arguments for bar plot
    n = len(som[0][0])
    coords = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    map = plt.get_cmap("rainbow")
    colors = [map(i / n) for i in range(1, n + 1)]

    # divide image into a grid to show the util nodes
    fig = plt.figure(figsize=(11, 8))
    grid = GridSpec(xdim, ydim)
    for x in range(xdim):
        for y in range(ydim):
            data = [i if i > 0 else 0 for i in som[x][y]]
            plt.subplot(grid[x, y], aspect=1, polar=True)
            plt.xticks([])
            plt.yticks([])
            plt.bar(x=coords, height=data, width=width, color=colors)
    # make legend
    handles = []
    for i, color in enumerate(colors):
        patch = mpatches.Patch(color=color, label=labels[i])
        handles.append(patch)
    fig.legend(handles=handles, title="Markers", title_fontsize=15)
    plt.savefig("som.jpg")
    plt.show()


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

    # get colors for the edges of the nodes
    if clusters is not None:
        color_map = plt.cm.get_cmap('rainbow', np.max(clusters)+1)

    # get colors for the markers
    markers = len(som[0])
    map = plt.get_cmap("rainbow")
    colors = [map(i / markers) for i in range(1, markers + 1)]

    # plot the nodes
    for node in tree.nodes:
        node_weight = [i if i > 0 else 0 for i in som[node]]
        if clusters is not None:
            color = color_map(clusters[node])
            draw_nodes(node_weight, pos[node], ax, fig, colors=colors, color=color)
        else:
            draw_nodes(node_weight, pos[node], ax, fig, colors=colors)
    # make legend
    handles = []
    for i, color in enumerate(colors):
        patch = mpatches.Patch(color=color, label=labels[i])
        if clusters is not None:
            patch = mpatches.Patch(color=color, label=f'Cluster {i}')
        handles.append(patch)
    fig.legend(handles=handles, loc="upper left",
               fontsize="20", title="Markers", title_fontsize=25)
    if clusters is not None:
        fig.legend(handles=handles, loc="upper left",
                   fontsize="20", title="Clusters", title_fontsize=25)
    # save the plot
    if clusters is None:
        plt.savefig("MSTNetworkX.jpg")
    else:
        plt.savefig("ClustersMSTNetworkX.jpg")
    plt.show()

def draw_nodes(data, pos, ax, fig, colors: list, color: str = "black"):
    piesize = 0.02
    xx, yy = ax.transData.transform(pos)  # figure coordinates
    xa, ya = fig.transFigure.inverted().transform((xx, yy))  # axes coordinates
    a = plt.axes([xa-0.01, ya-0.01, piesize, piesize], polar=True)
    a.axis("off")
    a.set_xticklabels([])
    a.set_yticklabels([])
    # prepare arguments for bar plot
    x = np.linspace(0.0, 2 * np.pi, len(data))
    width = 2 * np.pi / (len(data)+1)
    a.bar(x=x, height=data, width=width, color=colors, align="center")
    # place rectangle (that becomes circle) around the bar plot
    rect = mpatches.Rectangle(
        (0, np.max(data)*1.05), width=2*np.pi, height=np.max(data)/6,
        facecolor=color, linewidth=0
    )
    a.add_patch(rect)


def plot_MST_igraph(tree, som, labels, clusters=None):
    plt.figure(figsize=(15, 15))
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
    # make legend
    markers = len(som[0])
    map = plt.get_cmap("rainbow")
    colors = [map(i / markers) for i in range(1, markers + 1)]
    handles = []
    for i, color in enumerate(colors):
        patch = mpatches.Patch(color=color, label=labels[i])
        if clusters is not None:
            patch = mpatches.Patch(color=color, label=f'Cluster {i}')
        handles.append(patch)
    plt.legend(handles=handles, loc="upper left",
               fontsize=15, title="Markers", title_fontsize=20)
    if clusters is not None:
        plt.legend(handles=handles, loc="upper left",
                   fontsize=15, title="Clusters", title_fontsize=20)
    # save the plot
    if clusters is None:
        plt.savefig("MSTIGraph.jpg")
    else:
        plt.savefig("ClustersMSTIGraph.jpg")
    plt.show()


