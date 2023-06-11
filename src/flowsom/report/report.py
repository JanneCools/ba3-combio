from fpdf import FPDF


def write_intro(
    pdf: FPDF,
    size: int,
    colsToUse: list,
    xdim: int,
    ydim: int,
    dataset_name: str = None,
):
    """
    Write the intro of the report. This includes the title, dataset name and
    size, the used markers and the grid dimension.
    :param pdf: the pdf
    :param size: the size of the dataset
    :param colsToUse: the used columns (markers)
    :param xdim: x dimension of the grid
    :param ydim: y dimension of the grid
    :param dataset_name: the name of the dataset
    :return: None
    """
    pdf.set_font("Arial", "", 18)
    pdf.write(10, "FlowSOM algorithm\n\n")
    pdf.set_font("Arial", "", 14)
    if dataset_name is None:
        pdf.write(5, "Dataset: nameless\n\n")
    else:
        pdf.write(5, f"Dataset: {dataset_name}\n\n")
    pdf.write(5, f"Size dataset: {size} samples\n\n")
    pdf.write(5, f"Used markers: {', '.join(colsToUse)}\n\n")
    pdf.write(5, f"Grid dimensions: {xdim}x{ydim}\n\n\n")


def write_som(pdf: FPDF, minisom: bool, time: int):
    """
    Write information about the SOM to the pdf. This includes the used algorithm
    (library), its parameters and the training time.
    :param pdf: the pdf
    :param minisom: whether MiniSom or Sklearn-som is used
    :param time: the training time
    :return: None
    """
    pdf.set_font("Arial", "", 14)
    pdf.write(10, "Self organising map (SOM) of the dataset\n")
    pdf.set_font("Arial", "", 10)
    if minisom:
        pdf.write(5, "Used SOM-library: MiniSOM\n")
    else:
        pdf.write(5, "Used SOM-library: sklearn-som\n")
    pdf.write(5, f"Learning rate: 0.05\n")
    pdf.write(
        5,
        "Radius: 0,67 quantile of all neighbouring distances, using the Chebyshev metric\n",
    )
    pdf.write(5, f"Training time: {time} seconds\n")
    pdf.write(5, "Produced SOM-clusters:\n")
    pdf.image("som.jpg", w=90, h=70)


def write_mst(pdf: FPDF, networkx: bool, igraph: bool):
    """
    Write information about the MST to the pdf. This includes the used library
    and the MST itself.
    :param pdf: the pdf
    :param networkx: whether NetworkX is used
    :param igraph: whether IGraph is used
    :return: None
    """
    if networkx:
        pdf.write(5, "Minimal spanning tree of the SOM with NetworkX:\n")
        pdf.image("MSTNetworkX.jpg", w=90, h=90)
    if igraph:
        pdf.write(5, "Minimal spanning tree of the SOM with IGraph:\n")
        pdf.image("MSTIGraph.jpg", w=90, h=90)


def write_metaclustering(
    pdf: FPDF, n_clusters: int, networkx: bool, igraph: bool, time: int
):
    """
    Write information about the meta-clustering to the pdf. This includes the
    training time, amount of clusters and the distance metric.
    :param pdf: the pdf
    :param n_clusters: the amount of meta-clusters
    :param networkx: whether NetworkX is used
    :param igraph: whether IGraph is used
    :param time: the training time
    :return: None
    """
    pdf.set_font("Arial", "", 14)
    pdf.write(5, "Metaclusters of the SOM\n\n")
    pdf.set_font("Arial", "", 10)
    pdf.write(5, f"Amount of clusters for metaclustering: {n_clusters}\n")
    pdf.write(5, "Closest distance based on: average linking\n")
    pdf.write(5, f"Training time: {time} seconds\n")
    if networkx:
        pdf.write(5, "Visualisation using NetworkX:\n")
        pdf.image("ClustersMSTNetworkX.jpg", w=90, h=90)
    if igraph:
        pdf.write(5, "Visualisation using IGraph:\n")
        pdf.image("ClustersMSTIGraph.jpg", w=90, h=90)
