# @Desc  : Network indicators based on the trade brokerage property (TBP) module
import numpy as np
import pandas as pd
from tqdm import tqdm


def trade_brokerage_property(
        input_file_path, output_file_path, region_num, sector_num
):
    """
    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to the output XLSX file.
    :param region_num: Number of regions.
    :param sector_num: Number of sectors per region.
    """
    # Read the CSV file, using the first row as column names and the first column as the index.
    data = pd.read_csv(input_file_path, index_col=[0])
    # Check that the matrix size matches the specified region and sector counts.
    assert data.shape[0] == region_num * sector_num, "region_num或sector_num与输入数据不符"
    # Allocate the TBP results matrix.
    result = np.zeros((region_num * sector_num, 5))

    for t in tqdm(range(data.shape[0])):  # Iterate over the matrix nodes.
        temp_col = data.iloc[:, t]  # Column for node t.
        temp_row = data.iloc[t, :]  # Row for node t.
        rid = t // sector_num  # Region index; e.g., AUSS1–AUSS5 all belong to region 0.

        upstream = np.zeros(region_num)  # Intermediate inputs from upstream sectors, by region.
        for j in range(region_num):
            upstream[j] = np.sum(temp_col[j * sector_num: (j + 1) * sector_num])
        downstream = np.zeros(region_num)  # Intermediate outputs to downstream sectors, by region.
        for j in range(region_num):
            downstream[j] = np.sum(temp_row[j * sector_num: (j + 1) * sector_num])

        # Add a small constant to avoid division by zero.
        # Calculate the upstream and downstream regional probability vectors.
        prob_upstream = upstream / (np.sum(upstream) + 0.00001)
        prob_downstream = downstream / (np.sum(downstream) + 0.00001)

        # Calculate the five TBP components.
        result[t][0] = prob_upstream[rid] * prob_downstream[rid]
        result[t][1] = np.sum(prob_upstream * prob_downstream) - result[t][0]
        result[t][2] = (1 - prob_upstream[rid]) * prob_downstream[rid]
        result[t][3] = prob_upstream[rid] * (1 - prob_downstream[rid])
        temp = np.zeros(region_num)
        for j in range(region_num):
            if j == rid:
                temp[j] = 0
            else:
                temp[j] = 1 - prob_downstream[j] - prob_downstream[rid]
        result[t][4] = np.sum(prob_upstream * temp)
    # Save the results.
    result = pd.DataFrame(result, columns=['TBP1', 'TBP2', 'TBP3', 'TBP4', 'TBP5'], index=data.index)
    result['IMS'] = result['TBP3']/(result['TBP1']+result['TBP3'])
    result['VSD'] = (result['TBP2']+result['TBP5'])
    result['IIDI'] = result['IMS']*100
    result['IPTI'] = result['VSD']*100
    data.replace([np.inf, -np.inf], 0, inplace=True)
    result.fillna(0, inplace=True)
    result.to_excel(output_file_path)


if __name__ == '__main__':
    """
    Notes:
    1. The input must be a CSV file; the output is an XLSX file.
    2. The output directory must already exist.
    """
    input_file = "path/to/adjacency_matrix.csv"
    output_file = "path/to/tbp_results.xlsx"
    RegionNum = 63
    SectorNum = 35
    trade_brokerage_property(input_file, output_file, region_num=RegionNum, sector_num=SectorNum)
