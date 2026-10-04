# @Date  : 2024/11/10
# @Desc  : Network indicators based on the SWOT analysis module
import numpy as np
import pandas as pd
from tqdm import tqdm

np.set_printoptions(suppress=True)

def RAP(matrix):
    n = matrix.shape[0]
    h = np.tile(matrix.sum(axis=1), (n, 1))
    t = 1 / matrix.sum(axis=0)
    t[np.isinf(t)] = 0
    temp = np.empty([n, n], dtype=np.float32)
    for i in tqdm(range(n)):
        temp1 = np.tile(matrix[i, :], (n, 1))
        temp1 = np.multiply(temp1, matrix)
        temp[i, :] = np.dot(temp1, t)
        temp[i, i] = 0
    return temp/h

def GIRCN_and_Indices_from_GIVCN(matrix):
    GIRCN = RAP(matrix.T)
    GIRCN[np.isnan(GIRCN)] = 0
    SPOUT = GIRCN.sum(axis=1)
    SPIN = GIRCN.sum(axis=0)

    GPCCN = RAP(matrix)
    GPCCN[np.isnan(GPCCN)] = 0
    SOOUT = GPCCN.sum(axis=1)
    SOIN = GPCCN.sum(axis=0)

    ICWI = SPIN*100
    ICSI = SOIN*100

    return pd.DataFrame({
        'SPOUT': SPOUT,
        'SPIN': SPIN,
        'SOOUT': SOOUT,
        'SOIN': SOIN,
        'ICWI': ICWI,
        'ICSI': ICSI,
    })

def main(input_f, output_f):
    """
    :param input_f: Path to the input CSV file.
    :param output_f: Path to the output XLSX file.
    :return: None. The function writes an Excel workbook.
    """
    # Read the network matrix from CSV.
    data = pd.read_csv(input_f, index_col=[0])
    net = data.values
    # Compute the network indicators from the input matrix.
    result = GIRCN_and_Indices_from_GIVCN(net)
    result.index = data.index
    # Save the results as an Excel workbook.
    result.to_excel(output_f)

if __name__ == '__main__':
    """
    Notes:
    1. The input must be a CSV file; the output is an XLSX file.
    2. The output directory must already exist.
    """
    input_filename = r"D:\3.数据库\邻接矩阵\[GCFPN] ADB2025(E72) 73R35S CSV格式\2018.csv"
    output_filename = r"C:\Users\lenovo\Desktop\GIVCN-ADB73R35S\基于SWOT模块的计算结果\2018 SWOT Results.xlsx"
    main(input_filename, output_filename)
