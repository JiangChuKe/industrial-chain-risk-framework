# @Desc  : Random-walk network indicator: Random-walk centrality
import numpy as np
from tqdm import tqdm
import pandas as pd
np.set_printoptions(suppress=True)

def range_transform(matrix):
    result = matrix.copy()
    max_value = np.max(result)
    nonzeroresult = result[result != 0]
    min_value = np.min(nonzeroresult)
    result = (max_value - result) / (max_value - min_value)
    result[result < 0] = 0
    # Mark IFAI as missing where CRC is zero.
    result[matrix == 0] = np.nan
    return 100 * result

def Random_Walk_Centrality(net):
    N = len(net)
    net[net < 0] = 0
    np.round(net, 6)
    in_degree = net.sum(axis=0)
    in_zero = np.where(in_degree == 0)[0]
    net = np.delete(net, in_zero, axis=0)
    net = np.delete(net, in_zero, axis=1)
    n = len(net)
    net = net + 0.000001
    H = np.zeros(shape=(n, n))
    d = np.diag(np.sum(net, axis=1))
    M = np.dot(np.linalg.inv(d), net)
    net = np.eye(n) - M
    I = np.linalg.inv(net[1:, 1:])
    for i in tqdm(range(n)):
        index = list(range(n))
        H[[index[: i] + index[i + 1:]], [[i]]] = np.dot(I, np.ones(n - 1).T)
        if i < n - 1:
            I = np.linalg.inv(net[index[: i + 1] + index[i + 2:]][:, index[: i + 1] + index[i + 2:]])
    CRC = np.zeros(N)
    CRC[:n] = list(n / sum(H))
    for i in range(len(in_zero)):
        CRC[in_zero[i] + 1:] = CRC[in_zero[i]:N - 1]
        CRC[in_zero[i]] = 0
    CRC = np.nan_to_num(CRC)  # Replace NaN with zero and infinities with finite limits.
    # Apply range normalization.
    IFAI = range_transform(CRC)
    return pd.DataFrame({
        'CRC': CRC,
        'IFAI': IFAI.flatten(),
    })

def main(input_f, output_f):
    """
    :param input_f: Path to the input CSV file.
    :param output_f: Path to the output XLSX file.
    :return: None. The function writes an Excel workbook.
    """
    # Read the network matrix from CSV, using the first column as the index.
    net_df = pd.read_csv(input_f, index_col=0)
    print(net_df.shape)
    net = net_df.values
    # Compute random-walk centrality and its derived indicator.
    df = Random_Walk_Centrality(net)
    # Save the results as an Excel workbook.
    df.index = net_df.index
    df.to_excel(output_f)

if __name__ == '__main__':
    """
    Notes:
    1. The input must be a CSV file; the output is an XLSX file.
    2. The output directory must already exist.
    """
    input_filename = "path/to/adjacency_matrix.csv"
    output_filename = "path/to/crc_results.xlsx"

    main(input_filename, output_filename)
