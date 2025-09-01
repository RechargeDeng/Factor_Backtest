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

import update
import pandas as pd
import sys
sys.path.append('C:\\Users\\boyu.deng\\Desktop\\d1\\high_frec\\factor_test')
from DataLoader.DataLoader import DataLoader
from DataProcess.DataProcessor_3 import DataProcessor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import sys

from hfp.hfp import HFP
def main():
    data_path = 'C:\\Users\\boyu.deng\\Desktop\\d1\\hfrec_data'  # 原始数据的路径
    stock_num = 1  
    hfp = HFP(data_path=data_path, stock_num=stock_num,
         stock_list= ['BTCUSDT'], start_date='2025-01-01', end_date='2025-07-31', interval='1m')
    