# Copyright 2025 Jason Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
AutoTester
该模块用于测试信号对于高频价差的线性预测力
"""

import numpy as np


class Stats:
    def __init__(self, corr: np.array = None, mean_corr: float = 0, corr_IR: float = 0,
                 positive_corr_ratio: float = 0):
        """
        :param corr: 相关系数
        :param mean_corr: 平均相关系数
        :param corr_IR:
        :param positive_corr_ratio: 相关系数为正的比例
        """
        self.corr = corr
        self.mean_corr = mean_corr
        self.corr_IR = corr_IR
        self.positive_corr_ratio = positive_corr_ratio


class AutoTester:
    def __init__(self):
        pass

    @staticmethod
    def test(signal: np.array, ret: np.array, start: int = 100, end: int = 4600, shift: int = 1) -> Stats:
        """
        :param signal: 信号矩阵
        :param ret: 收益率矩阵
        :param start: 开始时间
        :param end: 结束时间
        :param shift: 预测平移量
        :return:
        """
        signal[np.isnan(signal)] = 0
        ret[np.isnan(ret)] = 0

        corr = np.zeros(signal.shape[1])

        for i in range(signal.shape[1]):
            corr[i] = np.corrcoef(ret[start+shift:end+shift, i], signal[start:end, i])[0, 1]
        mean_corr = np.nanmean(corr)
        corr_IR = mean_corr / np.nanstd(corr)
        positive_corr_ratio = np.sum(corr > 0) / np.sum(~np.isnan(corr))
        return (Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio), corr)

  
    # @staticmethod
    # def test(signal: np.array, ret: np.array, start: int = 100, end: int = 4600, shift: int = 1) -> Stats:
    #     """
    #     :param signal: 信号矩阵
    #     :param ret: 收益率矩阵
    #     :param start: 开始时间
    #     :param end: 结束时间
    #     :param shift: 预测平移量
    #     :return:
    #     """
    #     signal[np.isnan(signal)] = 0
    #     ret[np.isnan(ret)] = 0

    #     corr = np.zeros(signal.shape[1])
        
    #     # signal_1d = signal.reshape(-1)
    #     # ret_1d = ret.reshape(-1)
    #     # ret_1d_mean = ret_1d.mean()
    #     # ret_1d_std = ret_1d.std()
    #     # lower_bound = ret_1d_mean - 3 * ret_1d_std
    #     # upper_bound = ret_1d_mean + 3 * ret_1d_std
    #     # valid_indices = (ret_1d >= upper_bound) | (ret_1d <= lower_bound)
    #     # filtered_ret = ret_1d[valid_indices]
    #     # #print(len(filtered_ret))
    #     # filtered_signal = signal_1d[valid_indices]
    #     # corr = np.corrcoef(filtered_ret, filtered_signal)[0, 1]
    #     # mean_corr = corr
    #     # corr_IR = 0
    #     # positive_corr_ratio = 0
    #     # return (Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio), corr)

    #     for i in range(signal.shape[1]):
    #         # 获取ret的当前列数据
    #         ret_col = ret[start+shift:end+shift, i]
    #         signal_col = signal[start:end, i]
            
    #         # 计算ret列的均值和标准差
    #         ret_mean = np.mean(ret_col)
    #         ret_std = np.std(ret_col)
    #         #print(ret_mean, ret_std)
    #         # 寻找位于mean+3*std和mean-3*std之间的值的索引
    #         lower_bound = ret_mean - 3 * ret_std
    #         upper_bound = ret_mean + 3 * ret_std
    #         valid_indices = (ret_col >= upper_bound) | (ret_col <= lower_bound)
            
    #         # 获取对应索引位置的ret和signal值
    #         filtered_ret = ret_col[valid_indices]
    #         filtered_signal = signal_col[valid_indices]
    #         print(len(filtered_ret))
    #         # 计算相关系数
    #         if len(filtered_ret) > 1 and len(filtered_signal) > 1:
    #             corr[i] = np.corrcoef(filtered_ret, filtered_signal)[0, 1]
    #         else:
    #             corr[i] = np.nan
                
    #     mean_corr = np.nanmean(corr)
    #     corr_IR = mean_corr / np.nanstd(corr)
    #     positive_corr_ratio = np.sum(corr > 0) / np.sum(~np.isnan(corr))
    #     return (Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio), corr)