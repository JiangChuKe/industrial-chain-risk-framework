# @Date  : 2024/11/06
# @Desc  : Random-walk network indicator: Counting first-passage betweenness
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import os

np.set_printoptions(suppress=True)
dtype = torch.float32

def range_transform(matrix):
    result = matrix.copy()
    max_value = np.max(result)
    nonzeroresult = result[result != 0]
    min_value = np.min(nonzeroresult)
    result = (max_value - result) / (max_value - min_value)
    result[result < 0] = 0
    # Mark ICRI as missing where CFP is zero.
    result[matrix == 0] = np.nan
    return 100 * result

def Counting_First_Passage_Betweenness(net):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
        quit()

    N = len(net)
    net[net < 0] = 0
    in_degree = net.sum(axis=0)
    in_zero = np.where(in_degree == 0)[0]
    net = np.delete(net, in_zero, axis=0)
    net = np.delete(net, in_zero, axis=1)
    n = len(net)
    net = net + 0.000001
    take = 0.5 * ((net > 0) + (net.T > 0)).astype(float)
    d = np.diag(np.sum(net, axis=1))

    net = torch.tensor(net, dtype=dtype).to(device)
    d = torch.tensor(d, dtype=dtype).to(device)
    take = torch.tensor(take, dtype=dtype).to(device)
    re = torch.zeros(n, dtype=dtype, device=device)

    for t in tqdm(range(n), ncols=60):
        ind = [i for i in range(n) if i != t]
        ind = torch.tensor(ind, device=device)
        atemp_ = net[ind][:, ind]
        dtemp_ = d[ind][:, ind]
        T_ = torch.linalg.inv(dtemp_ - atemp_)

        for s in range(len(ind)):
            N_ = torch.diag(T_[s]) @ atemp_
            I = (N_ + N_.T).abs() * take[ind][:, ind]
            re[ind] += 0.5 * (I.sum(dim=0) + I.sum(dim=1))

    CFP = np.zeros(N)
    CFP[:n] = list((re.cpu().numpy() + 2 * (n - 1)) / (n * (n - 1)))

    for iz in in_zero:
        CFP[iz + 1:] = CFP[iz:N - 1]
        CFP[iz] = 0

    CFP = np.nan_to_num(CFP)  # Replace NaN with zero and infinities with finite limits.
    ICRI = range_transform(CFP)

    return pd.DataFrame({'CFP': CFP, 'ICRI': ICRI})

def main(input_f, output_f):
    """
    :param input_f: Path to the input CSV file.
    :param output_f: Path to the output XLSX file.
    :return: None. The function writes an Excel workbook.
    """
    # Read the network matrix from CSV, using the first column as the index.
    net_df = pd.read_csv(input_f, index_col=[0])
    net = net_df.values
    # Compute counting first-passage betweenness and its derived indicator.
    df = Counting_First_Passage_Betweenness(net)
    # Save the results as an Excel workbook.
    df.index = net_df.index
    df.to_excel(output_f)

if __name__ == '__main__':
    """
    Notes:
    1. The input must be a CSV file; the output is an XLSX file.
    2. The output directory must already exist.
    3. Install PyTorch with a CUDA version compatible with the available GPU:
       https://pytorch.org/get-started/locally/
    """
    input_filename = r"D:\3.数据库\邻接矩阵\CSV\[GIVCN] ADB2024(E62) 63R13S CSV格式\GIVCN-ADB(13S)-2021.csv"
    output_filename = r"C:\Users\lenovo\Desktop\2021 CFP Results.xlsx"

    main(input_filename, output_filename)
