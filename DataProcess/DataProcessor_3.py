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


import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob


class DataProcessor:
    def __init__(self, path=None, interval=None):
        """
        初始化数据处理器
        :param path: 数据文件路径，默认为'/Users/deng/Desktop/Dai/hfrec_data/BTCUSDT_futures_klines_csv/1d'
        """
        if path is None:
            path = "/Users/deng/Desktop/Dai/hfrec_data/BTCUSDT_futures_klines_csv"
        if path:
            path = os.path.join(path, f"{interval}")
        self.path = path
        self.interval = interval
    
    def process_data(self):
        """
        处理数据的主要方法
        1. 读取所有csv文件
        2. 按日期排序并拼接
        3. 计算三天收益率
        4. 重新保存文件
        """
        # 获取所有csv文件
        csv_files = glob.glob(os.path.join(self.path, "*.csv"))
        
        if not csv_files:
            print("未找到任何CSV文件")
            return
        def extract_date_from_filename(file_path):
            filename = os.path.basename(file_path)
            date_str = '-'.join(filename.split('-')[-3:]).replace('.csv', '')
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                return datetime(1900, 1, 1)

        csv_files.sort(key=extract_date_from_filename)
        # 读取所有文件并提取日期信息
        data_list = []
        for file_path in csv_files:
            # 从文件名中提取日期
            filename = os.path.basename(file_path)
            if len(filename.split('-')) > 5:
                continue
            date_str = '-'.join(filename.split('-')[-3:]).replace('.csv', '')
            #print(date_str)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            df['file_date'] = date_str
            data_list.append(df)
        
        # 按日期排序并拼接
        combined_df = pd.concat(data_list, ignore_index=True)
        combined_df = combined_df.sort_values(['file_date', 'close_time']).reset_index(drop=True)
        
        # 计算三天收益率
        combined_df['ret'] = combined_df['close'].pct_change(periods=3)#########################################
        try:
            combined_df['open_time_ms'] = combined_df['open_time']
            combined_df['open_time'] = pd.to_datetime(combined_df['open_time'], unit='ms')
        except:
            pass
        try:
            combined_df['close_time_ms'] = combined_df['close_time']
            combined_df['close_time'] = pd.to_datetime(combined_df['close_time'], unit='ms')
        except:
            pass

        #添加新的特征列
        combined_df['amplitude'] = combined_df['high'] - combined_df['low']
        combined_df['intra_ret'] = combined_df['close']/combined_df['open'] - 1
        combined_df['volume_ratio'] = combined_df['volume']/combined_df['quote_volume']
        combined_df['taker_buy_ratio'] = combined_df['taker_buy_volume']/combined_df['volume']
        combined_df['vwap'] = combined_df['quote_volume']/combined_df['volume']
        combined_df['avg_trade_size'] = combined_df['volume']/combined_df['count']
        combined_df['avg_trade_quote'] = combined_df['quote_volume']/combined_df['count']
        combined_df['close_vwap_vesus'] = combined_df['close'] - combined_df['vwap']


        # 按原始文件分解并保存
        for date in combined_df['file_date'].unique():
            date_df = combined_df[combined_df['file_date'] == date].copy()
            #print(date_df)
            date_df = date_df.drop('file_date', axis=1)  # 移除临时添加的日期列
            
            # 构造原始文件名
            filename = f"BTCUSDT-{self.interval}-{date}.csv"
            output_path = os.path.join(self.path, filename)
            
            # 保存文件
            date_df.to_csv(output_path, index=False)
            print(f"已保存文件: {filename}")
        
        print("数据处理完成")
        return combined_df
