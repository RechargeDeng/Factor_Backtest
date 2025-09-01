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
该模块定义读取高频数据的方法
"""


import numpy as np
import pandas as pd
import os


class Data:  # 存放一只股票的所有交易日的日内高频数据
    def __init__(self, data_dic: dict, ret: np.array):
        """
        :param data_dic: 数据字典，是一个
        :param ret: 收益率
        """
        self.data_dic = data_dic
        self.ret = ret


class DataLoader:
    def __init__(self, data_path: str = '/Users/deng/Desktop/Dai/hfrec_data',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData',
                 start_date = None, end_date = None, interval = None):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def load(self, stock_num: int = 1, stock_list: list = None) -> dict:
        """
        :param stock_num: 需要研究的股票数量
        :param stock_list: 需要研究的股票列表
        :return: dict，为所有股票的字典，值是Data
        """
        if stock_list is None:
            stock_list = os.listdir('{}/data'.format(self.data_path))#######
            stock_list = sorted(stock_list)
            stock_list = stock_list[:stock_num]
        datas = {}

        # names = ['BidSize1', 'BidSize2', 'BidSize3', 'BidSize4', 'BidSize5',
        #          'BidPX1', 'BidPX2', 'BidPX3', 'BidPX4', 'BidPX5',
        #          'OfferSize1', 'OfferSize2', 'OfferSize3', 'OfferSize4', 'OfferSize5',
        #          'OfferPX1', 'OfferPX2', 'OfferPX3', 'OfferPX4', 'OfferPX5',
        #          'mid_price', 'OrderCount', 'TradeCount', 'ret']
        names = ['open_time','open','high','low','close','volume','close_time','quote_volume','count',
                'taker_buy_volume','taker_buy_quote_volume','ignore','ret','amplitude','intra_ret','volume_ratio',
                'taker_buy_ratio','vwap','avg_trade_size','avg_trade_quote','close_vwap_vesus']
        try:
            date_list = pd.date_range(self.start_date, self.end_date, freq='D').strftime('%Y-%m-%d').tolist()
        except Exception as e:
            print(e)
        for coin in stock_list:
            stock_data = {name: [] for name in names}
            ret = []
            intervals = [self.interval] #os.listdir('{}/{}_futures_klines_csv'.format(self.data_path, coin))
            for interval in intervals:
                for date in date_list:
                    snapshot = pd.read_csv('{}/{}_futures_klines_csv/{}/{}-{}-{}.csv'.format(self.data_path, coin, interval ,coin, interval, date))
                    ret.append(snapshot['ret'])
                    for name in names:
                        stock_data[name].append(snapshot[name].values)
            #change
            max_len_ret = max(len(x) for x in ret)
            padded_ret = []
            for lst in ret:
                if len(lst) < max_len_ret:
                    # 用NaN或0填充
                    padded_lst = lst.tolist() + [np.nan] * (max_len_ret - len(lst))
                else:
                    padded_lst = lst
                padded_ret.append(padded_lst)
            ret = np.vstack(padded_ret).T

            for name in names:
                max_len = max(len(x) for x in stock_data[name])
                padded_stock_data = []
                for lst in stock_data[name]:
                    if len(lst) < max_len and name == 'close_time':
                        padded_lst = lst.tolist() + [np.nan] * (max_len - len(lst))
                    elif len(lst) < max_len and name != 'close_time':
                        padded_lst = lst.tolist() + [0] * (max_len - len(lst))
                    else:
                        padded_lst = lst
                    padded_stock_data.append(padded_lst)
                stock_data[name] = np.vstack(padded_stock_data).T
            datas[coin] = Data(data_dic=stock_data, ret=ret)
        return datas


        # for stock in stock_list:
        #     stock_data = {name: [] for name in names}  # 最终得到一个snapshot_num * days的矩阵，逻辑是每日可以看成是独立样本
        #     ret = []
        #     years = os.listdir('{}/data/{}'.format(self.data_path, stock))
        #     for year in years:
        #         days = os.listdir('{}/data/{}/{}'.format(self.data_path, stock, year))
        #         for day in days:
        #             snapshot = pd.read_csv('{}/data/{}/{}/{}/snapshot.csv'.format(self.data_path, stock, year, day))
        #             ret.append(snapshot['ret'])  # 3s收益率
        #             for name in names:
        #                 stock_data[name].append(snapshot[name].values)
        #     ret = np.vstack(ret).T
        #     for name in names:
        #         stock_data[name] = np.vstack(stock_data[name]).T
        #     datas[stock] = Data(data_dic=stock_data, ret=ret)
        # return datas