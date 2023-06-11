import os
from anndata import AnnData, concat
import readfcs
import numpy as np
import pandas


def read_directory(path: str) -> AnnData:
    """
    Read all files in the specified directory and create one AnnData object.
    The files must have the same markers, otherwise an Error will be thrown.
    :param path: the path to the directory
    :return: the AnnData object of all files in the directory
    """
    files = os.listdir(path)
    if len(files) == 0:
        raise AssertionError("Directory cannot be empty")
    datas = []
    # combine all AnnData objects
    for file in files:
        data = readfcs.read(path + "/" + file)
        if len(datas) != 0 and not np.array_equal(data.var_names, datas[0].var_names):
            raise AssertionError("All files must have the same markers")
        datas.append(data)
    adata = concat(datas)
    return adata


def remove_unused_data(adata: AnnData, columns: list) -> AnnData:
    """
    Removes the column of the AnnData object that are not used.
    :param adata: the AnnData object
    :param columns: the column that are used
    :return: the same AnnData object where the uns-object is extended with the
    filtered data
    """
    cols = adata.uns["meta"]["channels"]["$PnN"]
    indices = cols.index.astype(np.intc)
    # get the indices of the unused columns
    unused = [indices[i] for i, col in enumerate(cols) if col not in columns]
    unused = np.flip(np.sort(unused))
    data = adata.X
    # remove all unused columns
    for index in unused:
        data = np.delete(data, index-1, axis=1)
    # save the remaining data in the AnnData
    adata.uns["used_data"] = data
    return adata


def read_input(inp, colsToUse: list, adata: AnnData = None) -> AnnData:
    """
    Read the input and convert it to an AnnData object
    :param inp: the input, the type has multiple possibilities
    :param colsToUse: the columns that are used
    :param adata: an AnnData object
    :return: an AnnData object
    """
    # make anndata object
    if isinstance(inp, str):
        if inp[-4:] == ".fcs":
            adata = readfcs.read(inp)
        else:
            adata = read_directory(inp)
    elif isinstance(inp, np.ndarray) or isinstance(inp, pandas.DataFrame):
        adata = AnnData(inp)
    elif isinstance(int, AnnData):
        adata = inp
    if adata is None:
        raise ValueError("Input must be a file, directory, numpy object, pandas dataframe or an AnnData")

    # remove unused columns
    if colsToUse is None or "meta" not in adata.uns:
        adata.uns["used_data"] = adata.X
    else:
        adata = remove_unused_data(adata, colsToUse)
    return adata

