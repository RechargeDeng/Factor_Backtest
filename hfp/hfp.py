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
该模块定义快速测试特征的总类
"""


import numpy as np
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/High-Frequency-Predictor')
from DataLoader.DataLoader import DataLoader
from AutoFormula.AutoTester import AutoTester, Stats
#from AutoFormula.AutoFormula_cy import AutoFormula_cy
from AutoFormula.AutoFormula import AutoFormula


class HFP:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/HFData',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData',
                 stock_num: int = 1, stock_list = None, start_date = None, end_date = None, interval = None):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        :param stock_num: 测试股票数量
        :param stock_list: 测试股票列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        dl = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path,start_date=start_date,end_date=end_date,interval=interval)
        self.datas = dl.load(stock_num=stock_num, stock_list=stock_list)
        self.tester = AutoTester()
        self.auto_formula = {key: AutoFormula(value) for key, value in self.datas.items()}
        #self.auto_formula = {key: AutoFormula_cy(value) for key, value in self.datas.items()}##############

    def test_factor(self, formula: str, verbose: bool = True, start: int = 100,
                    end: int = 4600, shift: int = 1) -> dict:
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param verbose: 是否打印结果
        :param: start: 每日测试开始的snap
        :param: end: 每日测试结束的snap
        :param: shift: 预测平移量
        :return: 返回统计值以及该因子产生的信号矩阵的字典
        """
        to_return = {}
        for key, value in self.datas.items():
            stats_tuple, signal, signal_df = self.auto_formula[key].test_formula(formula, value, start=start, end=end, shift=shift)
            stats = stats_tuple[0]
            corr = stats_tuple[1]
            to_return[key] = (stats, signal, signal_df)
            if verbose:
                print('{} mean corr: {:.4f}, positive_corr_ratio: {:.4f}, corr_IR: {:.4f}'.
                      format(key, stats.mean_corr, stats.positive_corr_ratio, stats.corr_IR))
        return to_return, signal_df, corr