#!/usr/bin/env python3
# -*-coding:utf8 -*-
"""
Overview:
1. This script calculates four groups of SRPL-based network indicators: IEPI,
   IEUI_AND_IEDI, FISC_AND_IFDI, and BISC_AND_IBDI.
2. It reads a square CSV matrix, using the first column as row labels and the
   first row as column labels. It computes strongest relevance paths (SRPL) and
   their lengths, then saves the selected indicators to an Excel workbook named
   "<input filename> SRPL Results.xlsx".
3. When srpl_write=True, it also saves "<input filename> SRPL_Path.csv" and
   "<input filename> SRPL_Length.csv".
4. The output directory must already exist.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def range_transform(matrix: np.array, method):
    """
    Apply range normalization using the smallest nonzero value and maximum.

    With method='min', larger inputs produce larger outputs. With method='max',
    larger inputs produce smaller outputs.
    :param matrix: NumPy array to normalize.
    :param method: Normalization direction, either 'min' or 'max'.
    :return: Normalized NumPy array.
    """
    result = matrix.copy()
    max_value = np.nanmax(result)
    nonzeroresult = result[result != 0]
    min_value = np.nanmin(nonzeroresult)
    if method == 'min':
        result = (result - min_value) / (max_value - min_value)
    elif method == 'max':
        result = (max_value - result) / (max_value - min_value)
    else:
        raise ValueError("Enter the value of method as 'min' or 'max'")
    result[result < 0] = 0
    return 100 * result


def SRPL(data: pd.DataFrame):
    """
    Calculate strongest relevance paths and their values using the Floyd algorithm.
    :param data: DataFrame with identical row and column labels in the same order.
    :return: A pair of DataFrames (path, path_length). Each path cell contains
        a node sequence, and the corresponding path_length cell contains its value.
    """
    # Check that row and column labels match in the same order.
    assert data.columns.equals(data.index), "Inconsistent row and column names of the matrix"

    path_length = {}
    path = {}
    for index in data.index:
        path_length[index] = {}
        path[index] = {}
        for column in data.columns:
            if index == column:
                path_length[index][column] = 0  # Exclude self-links on the diagonal.
                path[index][column] = '0'  # Represent absent paths with '0'.
            else:
                path_length[index][column] = data.loc[index, column]
                if path_length[index][column] == 0:
                    path[index][column] = '0'
                else:
                    path[index][column] = f'{index}-{column}'
    # Apply the Floyd algorithm to update strongest relevance paths.
    for t in tqdm(data.index, desc="计算 SRPL"):
        n = len(t)
        for index in data.index:
            for column in data.columns:
                if path_length[index][t] == 0 or path_length[t][column] == 0:
                    continue
                temp = path_length[index][t] * path_length[t][column] / (path_length[index][t] + path_length[t][column])
                if path_length[index][column] < temp:
                    path_length[index][column] = temp
                    # Join the two path segments, omitting the repeated intermediate node t.
                    path[index][column] = path[index][t] + path[t][column][n:]
    return pd.DataFrame(path).T, pd.DataFrame(path_length).T


def IEPI(path: pd.DataFrame, boundary: int):
    """
    Calculate the IEPI indicator.
    :param path: DataFrame of strongest relevance path strings.
    :param boundary: Node-label prefix length used for region comparisons.
    :return: DataFrame with 'CB', 'CB_RB', 'CB_RE', and 'IEPI' columns.
    """
    # Convert the DataFrame to a dictionary for iteration.
    path_dict = path.T.to_dict()
    indices = path_dict.keys()
    n = len(indices)
    nbc = dict(zip(indices, np.zeros(n)))  # Store node betweenness counts.
    cbd = dict(zip(indices, np.zeros(n)))  # Store within-region betweenness contributions.
    cbf = dict(zip(indices, np.zeros(n)))  # Store cross-region betweenness contributions.

    for i in tqdm(indices, desc="计算 IEPI"):
        for j in indices:
            if path_dict[i][j] == '0':
                continue
            inter_nodes = path_dict[i][j].split('-')[1: -1]
            for node in inter_nodes:
                nbc[node] += 1
                count_cbd = 0
                count_cbf = 0
                for t in inter_nodes:
                    if t[:boundary] == node[:boundary]:
                        count_cbd += 1
                    else:
                        count_cbf += 1
                cbd[node] += count_cbd / len(inter_nodes)
                cbf[node] += count_cbf / len(inter_nodes)
    r = pd.DataFrame({
        'CB': nbc, 'CB_RB': cbd, 'CB_RE': cbf
    })
    # Calculate IEPI = (CB_RE / CB) * 100; return NaN where CB is zero.
    r['IEPI'] = np.divide(r['CB_RE'], r['CB'], out=np.full_like(r['CB'], np.nan), where=(r['CB'] != 0)) * 100
    return r


def IEUI_AND_IEDI(path_length: pd.DataFrame, countries_number: int, sectors_number: int):
    """
    Calculate IEUI and IEDI indicators.
    :param path_length: DataFrame of strongest relevance path values.
    :param countries_number: Number of countries or regions.
    :param sectors_number: Number of sectors per country or region.
    :return: DataFrame with CCIN, CCIN_RB, CCIN_RE, CCOUT, CCOUT_RB,
        CCOUT_RE, IEUI, and IEDI columns.
    """
    length = path_length.values
    index = path_length.index
    n = len(index)

    dt_part = np.zeros((n, n))
    for cn in range(countries_number):
        dim_start = cn * sectors_number
        dim_end = (cn + 1) * sectors_number
        dt_part[dim_start:dim_end, dim_start:dim_end] = 1
    it_part = 1 - dt_part
    length[np.isnan(length)] = 0
    length_it = np.multiply(length, it_part)
    length_dt = np.multiply(length, dt_part)

    t = np.count_nonzero(length, axis=0).astype(float)
    c = np.count_nonzero(length, axis=1).astype(float)

    ccin_it = np.divide(np.sum(length_it, axis=0), t, out=np.zeros_like(t), where=(t != 0))
    ccout_it = np.divide(np.sum(length_it, axis=1), c, out=np.zeros_like(c), where=(c != 0))

    ccin_dt = np.divide(np.sum(length_dt, axis=0), t, out=np.zeros_like(t), where=(t != 0))
    ccout_dt = np.divide(np.sum(length_dt, axis=1), c, out=np.zeros_like(c), where=(c != 0))

    ccin = np.divide(np.sum(length, axis=0), t, out=np.zeros_like(t), where=(t != 0))
    ccout = np.divide(np.sum(length, axis=1), c, out=np.zeros_like(c), where=(c != 0))

    ieui = np.divide(ccin_it, ccin, out=np.full_like(ccin, np.nan), where=(ccin != 0)) * 100
    iedi = np.divide(ccout_it, ccout, out=np.full_like(ccout, np.nan), where=(ccout != 0)) * 100

    result = pd.DataFrame({
        'CCIN': ccin, 'CCIN_RB': ccin_dt, 'CCIN_RE': ccin_it,
        'CCOUT': ccout, 'CCOUT_RB': ccout_dt, 'CCOUT_RE': ccout_it,
        'IEUI': ieui, 'IEDI': iedi
    })
    result.index = index
    return result


def FISC_AND_IFDI(path_length: pd.DataFrame, countries_number: int, sectors_number: int):
    """
    Calculate FISC (forward integration silhouette coefficient) and IFDI
    (industrial forward decoupling index).
    :param path_length: DataFrame of strongest relevance path values.
    :param countries_number: Number of countries or regions.
    :param sectors_number: Number of sectors per country or region.
    :return: DataFrame with 'FISC' and 'IFDI' columns.
    """
    length = path_length.values
    index = path_length.index
    n = len(index)

    frm = np.zeros((n, countries_number))
    for cn in range(countries_number):
        dim_start = cn * sectors_number
        dim_end = (cn + 1) * sectors_number
        frm[:, cn] = np.sum(length[:, dim_start:dim_end], axis=1)

    fisc = np.zeros(n)
    for cn in range(countries_number):
        dim_start = cn * sectors_number
        dim_end = (cn + 1) * sectors_number
        dfcs = frm[dim_start:dim_end, cn]
        etcs = np.delete(frm, cn, axis=1)
        metcs = np.max(etcs[dim_start:dim_end, :], axis=1)
        divisor = np.maximum(dfcs, metcs, dtype=float)
        divisor[divisor == 0] = np.nan
        fisc[dim_start:dim_end] = (dfcs - metcs) / divisor
    ifdi = range_transform(fisc, method='min')
    result = pd.DataFrame({
        'FISC': fisc, 'IFDI': ifdi
    })
    result.index = index
    return result


def BISC_AND_IBDI(path_length: pd.DataFrame, countries_number: int, sectors_number: int):
    """
    Calculate BISC and IBDI indicators.
    :param path_length: DataFrame of strongest relevance path values.
    :param countries_number: Number of countries or regions.
    :param sectors_number: Number of sectors per country or region.
    :return: DataFrame with 'BISC' and 'IBDI' columns.
    """
    result = FISC_AND_IFDI(path_length.T, countries_number, sectors_number)
    result.columns = ['BISC', 'IBDI']
    return result


def calculate_network_index(
        filepath: str,
        savepath: str,
        index_list: list or tuple = ('IEPI', 'IEUI_AND_IEDI', 'FISC_AND_IFDI', 'BISC_AND_IBDI'),
        boundary=None,
        countries_number=None,
        sectors_number=None,
        srpl_write=False
):
    """
    Read a square CSV matrix, calculate the selected network indicators, and
    save them to "<input filename> SRPL Results.xlsx".

    :param filepath: Path to the CSV matrix; the first column contains row
        labels and the first row contains column labels.
    :param savepath: Existing directory for output files.
    :param index_list: Indicator groups to calculate: 'IEPI',
        'IEUI_AND_IEDI', 'FISC_AND_IFDI', and/or 'BISC_AND_IBDI'.
    :param boundary: Node-label prefix length used to identify regions for IEPI.
    :param countries_number: Number of countries or regions.
    :param sectors_number: Number of sectors per country or region.
    :param srpl_write: Whether to save the SRPL path and length matrices as CSV.
    """
    # Validate the requested indicators and required parameters.
    if 'IEPI' in index_list:
        assert boundary is not None, "Please enter the correct index boundary value."
    if set(index_list).intersection({"IEUI_AND_IEDI", "FISC_AND_IFDI", "BISC_AND_IBDI"}):
        assert countries_number is not None and sectors_number is not None, "Please enter the correct number of countries and sectors."
    for index in index_list:
        assert index in ("IEPI", "IEUI_AND_IEDI", "FISC_AND_IFDI",
                         "BISC_AND_IBDI"), f"The calculation method for metric {index} does not exist."

    # Read the CSV matrix, using the first column as row labels.
    data = pd.read_csv(filepath, index_col=[0], header=[0])
    assert data.shape[0] == data.shape[1], "The input matrix is not a square matrix."
    if countries_number and sectors_number:
        assert data.shape[
                   0] == countries_number * sectors_number, "Please enter the correct number of countries and sectors."

    # Extract the input filename without its extension.
    base_filename = os.path.splitext(os.path.basename(filepath))[0]

    print("Search Strongest Relevance Path ...")
    path, path_length = SRPL(data)
    if srpl_write:
        # Prefix each auxiliary output filename with the input filename.
        path.to_csv(os.path.join(savepath, f"{base_filename} SRPL_Path.csv"))
        path_length.to_csv(os.path.join(savepath, f"{base_filename} SRPL_Length.csv"))

    results = []
    for metric in index_list:
        if metric == 'IEPI':
            print("Calculate IEPI ...")
            results.append(IEPI(path, boundary))
        elif metric == 'IEUI_AND_IEDI':
            print("Calculate IEUI AND IEDI ...")
            results.append(IEUI_AND_IEDI(path_length, countries_number, sectors_number))
        elif metric == 'FISC_AND_IFDI':
            print("Calculate FISC AND IFDI ...")
            results.append(FISC_AND_IFDI(path_length, countries_number, sectors_number))
        elif metric == 'BISC_AND_IBDI':
            print("Calculate BISC AND IBDI ...")
            results.append(BISC_AND_IBDI(path_length, countries_number, sectors_number))
        else:
            pass

    # Combine the forward and backward integration results when both are selected.
    if 'FISC_AND_IFDI' in index_list and 'BISC_AND_IBDI' in index_list:
        index1 = index_list.index('FISC_AND_IFDI')
        index2 = index_list.index('BISC_AND_IBDI')
        temp1 = results[index1]
        temp2 = results[index2]
        results = [v for i, v in enumerate(results) if i not in [index1, index2]]
        results.append(pd.concat([temp1, temp2], axis=1)[['BISC', 'FISC', 'IBDI', 'IFDI']])

    # Add node labels as the first column and a blank separator as the last column.
    for r in results:
        r.insert(0, 'Sector', r.index)
        r.insert(r.shape[1], '', np.nan)

    result_final = pd.concat(results, axis=1)
    # Render missing values in selected indicators as the string 'NaN'.
    cols_to_convert = ['IEPI', 'IEUI', 'IEDI', 'BISC', 'FISC', 'IBDI', 'IFDI']
    if set(result_final.columns).intersection(cols_to_convert):
        result_final[cols_to_convert] = result_final[cols_to_convert].astype(str)
        result_final[cols_to_convert] = result_final[cols_to_convert].replace('nan', 'NaN')
    # Derive the output filename from the input filename.
    output_filename = f"{base_filename} SRPL Results.xlsx"

    # Save the combined results as an Excel workbook.
    result_final.to_excel(os.path.join(savepath, output_filename), index=False)
    print(f"计算完成，结果已保存至 {os.path.join(savepath, output_filename)}")


if __name__ == '__main__':
    """
    Notes:
    1. The input must be a CSV file; the output is an XLSX file. The output
       directory must already exist.
    2. Set filepath and savepath to match your environment.
    """
    filepath = "path/to/adjacency_matrix.csv"
    savepath = "path/to/output_directory"
    index_list = ['IEPI', 'IEUI_AND_IEDI', 'BISC_AND_IBDI', 'FISC_AND_IFDI']

    calculate_network_index(
        filepath,
        savepath,
        index_list=index_list,
        boundary=3,
        countries_number=63,  # Use country or region boundaries.
        sectors_number=35,
        srpl_write=True      # Set to True to export the SRPL path and length matrices.
    )
