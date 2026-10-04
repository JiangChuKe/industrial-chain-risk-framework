# !/usr/bin/env python3
# -*-coding:utf8 -*-
# @Author: XuFeng
# @Date: 2024/12/23
import numpy as np
import pandas as pd


class Topsis:
    def __init__(self, excel_data: pd.ExcelFile):
        self.excel_data = excel_data
        self.data = None
        self.index = None

    # Calculate indicator weights using the entropy-weight method.
    def entropy_weight(self):
        """Step 1: Apply min-max scaling, reversing CRC and CFP."""
        for index in self.data.columns:
            max_ = self.data[index].max()
            min_ = self.data[index].min()
            if index in ['CRC', 'CFP']:
                self.data[index] = (max_ - self.data[index]) / (max_ - min_)
            else:
                self.data[index] = (self.data[index] - min_) / (max_ - min_)
        self.data.to_excel('./data/standardized_data.xlsx')
        """Step 2: Calculate the column-normalized proportion matrix."""
        # Sum the values in each indicator column.
        sum_cols = self.data.sum(axis=0)
        # Calculate each observation's proportion of its indicator total.
        p_matrix = self.data.divide(sum_cols, axis=1)
        p_matrix.to_excel('./data/p_matrix.xlsx')
        """Step 3: Calculate entropy for each indicator."""
        k = 1/np.log(p_matrix.shape[0])
        e_value = -k * p_matrix.multiply(np.log(p_matrix+1e-10)).sum(axis=0)
        """Step 4: Calculate one minus entropy for each indicator."""
        d_value = 1 - e_value
        d_value.to_excel('./data/d_value.xlsx')
        """Step 5: Normalize the coefficients into indicator weights."""
        weight = d_value / d_value.sum()
        score = self.data.multiply(weight)
        self.data = score

    # Calculate relative closeness using TOPSIS.
    def topsis(self):
        """Step 1: Identify the positive and negative ideal values for each indicator."""
        max_ = self.data.max(axis=0)
        min_ = self.data.min(axis=0)
        max_.to_frame().to_excel('./data/max_value.xlsx')
        min_.to_frame().to_excel('./data/min_value.xlsx')
        """Step 2: Calculate each observation's Euclidean distance from both ideal solutions."""
        distance_max = self.data.apply(lambda x: np.linalg.norm(x - max_), axis=1)
        distance_min = self.data.apply(lambda x: np.linalg.norm(x - min_), axis=1)
        distance_max.to_frame().to_excel('./data/distance_max.xlsx')
        distance_min.to_frame().to_excel('./data/distance_min.xlsx')
        """Step 3: Calculate each observation's relative closeness to the positive ideal solution."""
        c = (distance_min / (distance_min + distance_max)).to_frame()
        c.columns = ['c']
        return c

    # Reshape the indicator sheets from wide to long panel format.
    def wide_to_long(self):
        # Reshape and combine all worksheets except the last one.
        results = []
        for index in self.excel_data.sheet_names[:-1]:
            data = self.excel_data.parse(index)
            if self.index is None:
                self.index = data['RegionSector']
            data = pd.melt(data, id_vars='RegionSector', var_name='year', value_name=index)
            data.set_index(['RegionSector', 'year'], inplace=True)
            results.append(data)
        results = pd.concat(results, axis=1)
        self.data = results

    # Reshape the results from long to wide panel format.
    def long_to_wide(self):
        self.data = self.data.pivot(index='RegionSector', columns='year', values='c')

    def main(self):
        # Parse the input workbook into a long panel.
        self.wide_to_long()
        # Apply entropy-derived indicator weights.
        self.entropy_weight()
        # Calculate TOPSIS relative closeness scores.
        c = self.topsis()
        self.data = c.reset_index()
        self.long_to_wide()
        # Save the results.
        self.data.loc[self.index, :].to_csv('./data/result.csv')


if __name__ == '__main__':
    file_name = './data/【用于Python】基于GIVCN-ADB模型的风险指标统计（ADB2025 35R63S 2007-2024）.xlsx'
    excel = pd.ExcelFile(file_name)
    topsis = Topsis(excel)
    topsis.main()
