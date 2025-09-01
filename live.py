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

import logging
import glob
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
import requests
import pandas as pd
import datetime
from datetime import timedelta
import time
import pickle
import multiprocessing
from multiprocessing import Pool
#import down_k
import os
from tqdm import tqdm
import update
import sys
sys.path.append('C:\\Users\\boyu.deng\\Desktop\\d1\\high_frec\\factor_test')
from DataLoader.DataLoader import DataLoader
from DataProcess.DataProcessor_3 import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
import importlib
from hfp.hfp import HFP
import update
config_logging(logging, logging.INFO)

#所有永续U本位合约
#投研时间段：2022-2025/03
# url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
# response = requests.get(url)
# data = response.json()

# 提取所有永续合约的symbol
# perpetual_symbols = [
#     s['symbol'] for s in data['symbols'] 
#     if (s['contractType'] == 'PERPETUAL') & (s['onboardDate']<=1640995200000) #onboard时间早于2022-01-01
# ]
perpetual_symbols = ['BTCUSDT']

def timestamp_convert(date_str):
    # 创建datetime对象
    # 将日期字符串转换为datetime对象
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    try:
        # 尝试解析带微秒的格式
        dt_obj = datetime.datetime.strptime(date_str, date_format)
    except ValueError:
        try:
            # 如果失败，尝试不带微秒的格式
            dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 如果还是失败，可能是其他格式，抛出异常
            raise ValueError(f"无法解析日期字符串: {date_str}")
    
    # 添加UTC时区信息
    dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
    timestamp_ms = int(dt_obj.timestamp() * 1000)
    #print(f"timestamp_ms after converting: {timestamp_ms}")
    return timestamp_ms

def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60, "w": 7 * 24 * 60 * 60}

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def antierror_get_interval_kline(symbol,contractType,interval,limit,startTime,endTime,um_futures_client):
    idx=1
    while True:
        try:
            temp_data = um_futures_client.continuous_klines(
                pair=symbol,
                contractType=contractType, #'PERPETUAL',
                interval=interval,
                limit=limit,
                startTime=startTime,
                endTime=endTime
            )
            connection_error= 0 if idx == 1 else 1
            return temp_data, connection_error  # 如果成功获取数据就返回并退出循环
        except Exception as e:
            print(f'handling {e}, please wait{idx}...')
            idx+=1
            time.sleep(20)  # 休眠20秒
            try:
                um_futures_client = UMFutures()  # 重新初始化客户端
                temp_data = um_futures_client.continuous_klines(
                    pair=symbol,
                    contractType=contractType, #'PERPETUAL',
                    interval=interval,
                    limit=limit,
                    startTime=startTime,
                    endTime=endTime
                )
                connection_error= 0 if idx == 1 else 1
                return temp_data, connection_error
            except:
                print(f'handling {e}, please wait{idx}...')
                idx+=1
                time.sleep(20)
                continue  # 如果初始化失败继续循环
        
        


def get_historical_klines(symbol, interval, start_str, end_str=None):
    # start_ts=timestamp_convert(start_str)
    # end_ts=timestamp_convert(end_str)
    start_ts=start_str
    end_ts=end_str
    limit=1500
    output_data = []
    connection_error=0

    try:
        um_futures_client = UMFutures()
    except Exception as e:
        print(f'handling {e}, please wait0...')
        connection_error+=1
        time.sleep(20)
        um_futures_client = UMFutures()

    timeframe = interval_to_milliseconds(interval)
    assert timeframe is not None

    idx = 0
    symbol_existed = False

    while True:
        temp_data, connection_error=antierror_get_interval_kline(symbol,'PERPETUAL',interval,limit,start_ts,end_ts,um_futures_client)

        # 处理首次请求：跳过交易对未上市时段
        if not symbol_existed and len(temp_data):
            symbol_existed = True
        
        if symbol_existed:
            output_data += temp_data  # 追加数据
            start_ts = temp_data[-1][0] + timeframe  # 更新下次请求的起始时间
        else:
            start_ts += timeframe  # 交易对未上市，直接推移时间窗口

        # 退出条件：返回数据不足500条（已到终点）
        if len(temp_data) < limit:
            break
        
        # 频率控制：每3次请求暂停1秒
        if idx % 3 == 0:
            time.sleep(1)
    return (output_data,symbol,connection_error)

#并行拿数据
# 创建进程池，使用可用CPU核心数
#processes=multiprocessing.cpu_count()//2

def down_data(symbol_list, frec, start_date, end_date, mark_date=None):
    for symbol in symbol_list:
        result = get_historical_klines(symbol,frec,start_date,end_date)
        result_df = pd.DataFrame(result[0], columns=[
            'open_time',         # 开盘时间
            'open',        # 开盘价
            'high',        # 最高价
            'low',         # 最低价
            'close',       # 收盘价
            'volume',            # 成交量
            'close_time',        # 收盘时间
            'quote_volume',      # 成交额
            'count',      # 成交笔数
            'taker_buy_volume',        # 主动买入成交量
            'taker_buy_quote_volume',  # 主动买入成交额
            'ignore'             # 忽略参数
        ])

        if not os.path.exists(f'C:/Users/boyu.deng/Desktop/d1/hfrec_data/{symbol}_futures_klines_csv/{frec}'):
            os.makedirs(f'C:/Users/boyu.deng/Desktop/d1/hfrec_data/{symbol}_futures_klines_csv/{frec}')
        result_df['ret'] = 0 #result_df['close'].pct_change(periods=3)#########################################
        try:
            result_df['open_time_ms'] = result_df['open_time']
            result_df['open_time'] = pd.to_datetime(result_df['open_time'], unit='ms')
        except:
            pass
        try:
            result_df['close_time_ms'] = result_df['close_time']
            result_df['close_time'] = pd.to_datetime(result_df['close_time'], unit='ms')
        except:
            pass
        # 将相关列转换为数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        print(type(result_df['high'].iloc[-1]))
        #添加新的特征列
        result_df['amplitude'] = result_df['high'] - result_df['low']
        result_df['intra_ret'] = result_df['close']/result_df['open'] - 1
        result_df['volume_ratio'] = result_df['volume']/result_df['quote_volume']
        result_df['taker_buy_ratio'] = result_df['taker_buy_volume']/result_df['volume']
        result_df['vwap'] = result_df['quote_volume']/result_df['volume']
        result_df['avg_trade_size'] = result_df['volume']/result_df['count']
        result_df['avg_trade_quote'] = result_df['quote_volume']/result_df['count']
        result_df['close_vwap_vesus'] = result_df['close'] - result_df['vwap']
        # result_df.to_csv(f'kline_{frec}/{symbol}_{frec}_{start_date}_{end_date}.csv')
        return result_df
        #result_df.to_csv(f'C:/Users/boyu.deng/Desktop/d1/hfrec_data/{symbol}_futures_klines_csv/{frec}/{symbol}-{frec}-{mark_date}.csv')
        #connection_error=result[2]

def paral_down_data(symbol_list, frec, start_date, end_date, processes):  #'1m','2022-01-01 00:00:00.000000','2022-12-31 23:59:59.000000'
    with Pool(processes=processes) as pool:
        # 并行执行下载任务
        results=[]
        # 收集报错
        error_count=[]
        for symbol in symbol_list:
            result = pool.apply_async(get_historical_klines,
                                            (symbol,frec,start_date,end_date,))
            result_get=result.get()
            result_df = pd.DataFrame(result_get[0], columns=[
                'open_time',         # 开盘时间
                'open_price',        # 开盘价
                'high_price',        # 最高价
                'low_price',         # 最低价
                'close_price',       # 收盘价
                'volume',            # 成交量
                'close_time',        # 收盘时间
                'quote_volume',      # 成交额
                'trades_count',      # 成交笔数
                'buy_volume',        # 主动买入成交量
                'buy_quote_volume',  # 主动买入成交额
                'ignore'             # 忽略参数
            ])

            if not os.path.exists(f'data/{symbol}'):
                os.makedirs(f'data/{symbol}')

            # result_df.to_csv(f'kline_{frec}/{symbol}_{frec}_{start_date}_{end_date}.csv')
            result_df.to_csv(f'data/{symbol}/{symbol}_{frec}_{start_date}_{end_date}.csv')
            connection_error=result_get[2]
            error_count.append((symbol,connection_error))
            # results.append(result)
            #print(symbol)
        pool.close()
        pool.join()
    print(error_count)

def data_process(file_path,interval):
    csv_files = glob.glob(os.path.join(file_path, "*.csv"))

    def extract_date_from_filename(file_path):
        filename = os.path.basename(file_path)
        date_str = '-'.join(filename.split('-')[-3:]).replace('.csv', '')
        try:
            return datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return datetime.datetime(1900, 1, 1)

    csv_files.sort(key=extract_date_from_filename)
    data_list = []
    for path in csv_files:
        filename = os.path.basename(path)
        date_str = '-'.join(filename.split('-')[-3:]).replace('.csv', '')
        df = pd.read_csv(path)
        df['file_date'] = date_str
        data_list.append(df)
    combined_df = pd.concat(data_list, ignore_index=True)
    combined_df['ret'] = combined_df['close'].pct_change(periods=288).shift(-288)
    # 按原始文件分解并保存
    
    for date in tqdm(combined_df['file_date'].unique(), desc="file processing"):
        date_df = combined_df[combined_df['file_date'] == date].copy()
        #print(date_df)
        date_df = date_df.drop('file_date', axis=1)  # 移除临时添加的日期列
        
        # 构造原始文件名
        filename = f"BTCUSDT-{interval}-{date}.csv"
        output_path = os.path.join(file_path, filename)
        
        # 保存文件
        date_df.to_csv(output_path, index=False)
        #print(f"已保存文件: {filename}")

if __name__=='__main__':
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='高频因子测试实时更新程序')
    parser.add_argument('--data_path', type=str, default='C:/Users/boyu.deng/Desktop/d1/hfrec_data',
                        help='原始数据的路径')
    parser.add_argument('--start_date', type=str, default='2025-01-01',
                        help='开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1m',
                        help='时间间隔 (如: 1m, 5m, 1h)')
    parser.add_argument('--update_interval', type=int, default=60,
                        help='更新间隔（秒）')
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用解析的参数
    data_path = args.data_path
    start_date = args.start_date
    # end_date = args.end_date
    interval = args.interval
    update_interval = args.update_interval
    
    print(f"Configuration parameters:")
    print(f"Data path: {data_path}")
    print(f"Start date: {start_date}")
    #print(f"  End date: {end_date}")
    print(f"Data time interval: {interval}")
    print(f"Update interval: {update_interval}s")
    datenow = pd.to_datetime(int(time.time()*1000), unit='ms').strftime('%Y-%m-%d')
    stock_num = 1  
    
    while True:
        start_t = pd.to_datetime(int(time.time()*1000), unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        print('-'*100)
        print(f'{start_t}data updating...')
        time_counter_start = time.time()
        print('-'*100)
        # 获取目录列表
        # dir_list = os.listdir('/Users/deng/Desktop/crypto_research/Data_download/kline_30m') ########
        #dir_list = os.listdir('/Users/deng/Desktop/crypto_research/Data_download/data') ########
        # 提取含有USDT的元素并截取USDT及之前的部分
        # usdt_symbols = [item.split('USDT')[0] + 'USDT' for item in dir_list if 'USDT' in item]
        path = f'{data_path}/{perpetual_symbols[0]}_futures_klines_csv/{interval}'#'C:/Users/boyu.deng/Desktop/d1/hfrec_data/BTCUSDT_futures_klines_csv/1m'
        timenow = pd.to_datetime(int(time.time()*1000), unit='ms').strftime('%Y-%m-%d')
        time_delay = (pd.to_datetime(timenow) - timedelta(days=1)).strftime('%Y-%m-%d')
        csv_files = glob.glob(os.path.join(path, "*.csv"))
            
        if not csv_files:
            print("未找到任何CSV文件")
        def extract_date_from_filename(file_path):
            filename = os.path.basename(file_path)
            date_str = '-'.join(filename.split('-')[-3:]).replace('.csv', '')
            try:
                return datetime.datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                return datetime.datetime(1900, 1, 1)

        csv_files.sort(key=extract_date_from_filename)
        last_file = pd.read_csv(csv_files[-1])
        last_file_len = len(last_file)
        file_len_bar= {
            '1m':1440,
            '3m':480,
            '5m':288,
            '1h':24
        }
        if last_file_len < file_len_bar[interval]:
            #last_close_time = last_file['close_time_ms'].iloc[0]
            strat_time = last_file['open_time_ms'].iloc[0]#last_close_time + 1
            end_time = int(time.time()*1000)
            mark_date = pd.to_datetime(strat_time, unit='ms').strftime('%Y-%m-%d')
            result_df = down_data(perpetual_symbols, interval, strat_time, end_time, mark_date)
            result_df.to_csv(f'{data_path}/{perpetual_symbols[0]}_futures_klines_csv/{interval}/BTCUSDT-{interval}-{mark_date}.csv')
        #更改这个条件下的逻辑，如果条件为真，则用down_data得到last_file['close_time_ms'].iloc[-1]+1开始，到现在时间戳结束的df
        elif last_file_len == file_len_bar[interval]:
            last_close_time = last_file['close_time_ms'].iloc[-1]
            start_time = last_close_time + 1
            end_time = int(time.time()*1000)
            
            # 使用down_data获取数据
            result_df = down_data(perpetual_symbols, interval, start_time, end_time, '')
            
            if not result_df.empty:
                # 添加日期列用于分组
                result_df['date'] = pd.to_datetime(result_df['close_time_ms'], unit='ms').dt.strftime('%Y-%m-%d')
                
                # 按日期分组并保存
                for date in result_df['date'].unique():
                    date_df = result_df[result_df['date'] == date].copy()
                    date_df = date_df.drop('date', axis=1)  # 移除临时日期列
                    
                    # 保存为CSV文件
                    output_path = f'{data_path}/{perpetual_symbols[0]}_futures_klines_csv/{interval}/BTCUSDT-{interval}-{date}.csv'
                    date_df.to_csv(output_path, index=False)
                    print(f"已保存文件: BTCUSDT-{interval}-{date}.csv")
            else:
                print("未获取到新数据")
        print('-'*100)
        time_counter_end = time.time()
        print(f'data update time: {time_counter_end - time_counter_start}s')
        print('-'*100)
        print('\n')
        print('-'*100)
        print(f'data processing...')
        time_counter_start = time.time()
        print('-'*100)
        data_process(path, interval)
        print('-'*100)
        time_counter_end = time.time()
        print(f'data processing time: {time_counter_end - time_counter_start}s')
        print('-'*100)

        print('\n')
        print('-'*100)
        print(f'signal generating...')
        time_counter_start = time.time()
        print('-'*100)
        hfp = HFP(data_path=data_path, stock_num=stock_num,
            stock_list= ['BTCUSDT'], start_date=start_date, end_date=timenow, interval=interval) ##########
        fml_list = ['tsregres{avg_trade_quote,avg_trade_size,720}',
            'tscov{quote_volume,volume_ratio,720}',
            'tscorr{taker_buy_quote_volume,volume_ratio,360}',
            'tscov{volume,volume_ratio,720}',
            'neg{tsmean{close,10}}',
            'tscorr{taker_buy_volume,volume_ratio,360}',
            'neg{tscov{quote_volume,taker_buy_ratio,720}}',
            'neg{tsregres{taker_buy_quote_volume,tspct{close,720},360}}',
            'neg{absv{close}}'
            ]
        #fml_list = ['tsregres{avg_trade_quote,avg_trade_size,60}']
        factor_stats = {name:[] for name in ['key', 'formula', 'mean_corr','positive_corr_ratio','corr_IR']}
        combined_signal_df = pd.DataFrame()
        combined_corr_df = pd.DataFrame()
        for fml in tqdm(fml_list[:], desc="Testing formula progress"):
            try:
                stats, signal_df, corr = hfp.test_factor(fml,verbose=False, shift=0, start = 10, end = 1400)  # shift表明对下几个三秒的收益率进行预测
            except Exception as e:
                continue

            # 将corr转换为DataFrame并添加时间列
            corr_df = pd.DataFrame({'corr': corr})
            
            # 生成从start_date到timenow的日期范围
            date_range = pd.date_range(start=start_date, end=timenow, freq='D')
            
            # 如果corr的长度与日期范围长度不匹配，调整日期范围
            if len(corr_df) != len(date_range):
                if len(corr_df) < len(date_range):
                    date_range = date_range[:len(corr_df)]
                else:
                    # 如果corr更长，重复最后的日期
                    additional_dates = pd.date_range(start=date_range[-1], periods=len(corr_df)-len(date_range)+1, freq='D')[1:]
                    date_range = date_range.append(additional_dates)
            
            # 添加时间列，格式为YYYY-MM-DD
            corr_df['time'] = date_range.strftime('%Y-%m-%d')
            corr_df.columns = [f'{fml}','time']
            if not combined_corr_df.empty:
                combined_corr_df = pd.merge(combined_corr_df, corr_df, on='time', how='inner')
            else:
                combined_corr_df = corr_df
            
            for key, value in stats.items():
                factor_stats['key'].append(key)
                factor_stats['formula'].append(fml)
                factor_stats['mean_corr'].append(value[0].mean_corr)
                factor_stats['positive_corr_ratio'].append(value[0].positive_corr_ratio)
                factor_stats['corr_IR'].append(value[0].corr_IR)
            # 调整signal_df的列名
            signal_df.columns = ['time', f'{fml}']
            signal_df_clean = signal_df.dropna()
            signal_df_clean = signal_df_clean[~signal_df_clean.isin(['NaN', 'nan']).any(axis=1)]
            # 对信号列进行归一化处理：(值-最小值)/(最大值-最小值)，然后减去0.5后乘以2
            signal_col = f'{fml}'
            if signal_col in signal_df_clean.columns and len(signal_df_clean[signal_col]) > 0:
                min_val = signal_df_clean[signal_col].min()
                max_val = signal_df_clean[signal_col].max()
                if max_val != min_val:  # 避免除零错误
                    signal_df_clean[signal_col] = ((signal_df_clean[signal_col] - min_val) / (max_val - min_val) - 0.5) * 2
                else:
                    signal_df_clean[signal_col] = 0  # 如果最大值等于最小值，设为0
            if not combined_signal_df.empty:
                combined_signal_df = pd.merge(combined_signal_df, signal_df_clean, on='time', how='inner')
            else:
                combined_signal_df = signal_df_clean

        combined_signal_df = combined_signal_df.dropna()
        # 将time列转换为datetime类型并设置为索引
        combined_signal_df['time'] = pd.to_datetime(combined_signal_df['time'])
        combined_signal_df = combined_signal_df.set_index('time')
        
        # 保存最后一行数据
        last_row = combined_signal_df.iloc[-1:].copy()
        
        # 重采样为5分钟频率，使用最后一个值
        resampled_df = combined_signal_df.resample('5T').last()
        
        # 确保最后一行数据被保留
        if last_row.index[0] not in resampled_df.index:
            resampled_df = pd.concat([resampled_df, last_row])
        
        # 重置索引，将time从索引转回列
        combined_signal_df = resampled_df.reset_index()
        combined_signal_df.to_csv(f'C:/Users/boyu.deng/Desktop/d1/signal_data/combined_df.csv')
        combined_corr_df.to_csv(f'C:/Users/boyu.deng/Desktop/d1/signal_data/combined_corr_df.csv')
        print('-'*100)
        time_counter_end = time.time()
        print(f'signal generating time: {time_counter_end - time_counter_start}s')
        print('-'*100)

        print('\n')
        print('-'*100)
        print(f'signal updating to GitHub...')
        update.update_once()
        print('-'*100)
        time_counter_end = time.time()
        print(f'signal updating to GitHub time: {time_counter_end - time_counter_start}s')
        print('-'*100)
        time.sleep(update_interval)

#python live.py --data_path "C:/Users/boyu.deng/Desktop/d1/hfrec_data" --start_date 2025-03-01 --interval 1m --update_interval 30
#