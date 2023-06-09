import os
import anndata
import readfcs
import numpy as np
import pandas


def read_directory(path: str):
    files = os.listdir(path)
    if len(files) == 0:
        raise AssertionError("Directory cannot be empty")
    datas = []
    for file in files:
        data = readfcs.read(path + "/" + file)
        if len(datas) != 0 and not np.array_equal(data.var_names, datas[0].var_names):
            raise AssertionError("All files must have the same markers")
        datas.append(data)
    adata = anndata.concat(datas)
    return adata


def remove_unused_data(adata: anndata.AnnData, columns: list):
    cols = adata.uns["meta"]["channels"]["$PnN"]
    indices = cols.index.astype(np.intc)
    unused = [indices[i] for i, col in enumerate(cols) if col not in columns]
    unused = np.flip(np.sort(unused))
    data = adata.X
    for index in unused:
        data = np.delete(data, index-1, axis=1)
    # only save the data from the used columns
    adata.uns["used_data"] = data
    return adata


def read_input(inp, colsToUse: list, adata=None):
    # make anndata object
    if isinstance(inp, str):
        if inp[-4:] == ".fcs":
            adata = readfcs.read(inp)
        else:
            adata = read_directory(inp)
    elif isinstance(inp, np.ndarray) or isinstance(inp, pandas.DataFrame):
        adata = anndata.AnnData(inp)
    elif isinstance(int, anndata.AnnData):
        adata = inp
    if adata is None:
        raise ValueError("Input must be a file, directory, numpy object, pandas dataframe or an AnnData")

    # remove unused columns
    if colsToUse is None or "meta" not in adata.uns:
        adata.uns["used_data"] = adata.X
    else:
        adata = remove_unused_data(adata, colsToUse)
    return adata

