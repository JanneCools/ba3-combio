import fpdf


def write_intro(
        pdf: fpdf.FPDF,
        size: int,
        colsToUse: list,
        xdim: int,
        ydim: int,
        dataset_name: str = None,
):
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


def write_som(pdf: fpdf.FPDF, minisom: bool, time: int):
    pdf.set_font("Arial", "", 14)
    pdf.write(10, "Self organising map (SOM) of the dataset\n")
    pdf.set_font("Arial", "", 10)
    if minisom:
        pdf.write(5, "Used SOM-library: MiniSOM\n")
    else:
        pdf.write(5, "Used SOM-library: sklearn-som\n")
    pdf.write(5, f"Learning rate: 0.05\n")
    pdf.write(5, "Radius: 0,67 quantile of all neighbouring distances, using the Chebyshev metric\n")
    pdf.write(5, f"Training time: {time} seconds\n")
    pdf.write(5, "Produced SOM-clusters:\n")
    pdf.image("som.jpg", w=90, h=70)


def write_mst(pdf: fpdf.FPDF, networkx: bool, igraph: bool):
    if networkx:
        pdf.write(5, "Minimal spanning tree of the SOM with NetworkX:\n")
        pdf.image("mstnetworkx.jpg", w=90, h=90)
    if igraph:
        pdf.write(5, "Minimal spanning tree of the SOM with IGraph:\n")
        pdf.image("mst_igraph.jpg", w=90, h=90)


def write_metaclustering(
        pdf: fpdf.FPDF,
        n_clusters: int,
        networkx: bool,
        igraph: bool,
        time: int
):
    pdf.set_font("Arial", "", 14)
    pdf.write(5, "Metaclusters of the SOM\n\n")
    pdf.set_font("Arial", "", 10)
    pdf.write(5, f"Amount of clusters for metaclustering: {n_clusters}\n")
    pdf.write(5, "Closest distance based on: average linking\n")
    pdf.write(5, f"Training time: {time} seconds\n")
    if networkx:
        pdf.write(5, "Visualisation using NetworkX:\n")
        pdf.image("clustersmstnetworkx.jpg", w=90, h=90)
    if igraph:
        pdf.write(5, "Visualisation using IGraph:\n")
        pdf.image("clusters_mst_igraph.jpg", w=90, h=90)
