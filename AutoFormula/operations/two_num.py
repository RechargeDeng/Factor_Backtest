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
该代码定义2_num_num型运算符
"""


import numpy as np
import numba as nb


# @nb.jit
def tsregres(a, b, num):  # 回溯num天时序回归残差
    s = np.zeros(a.shape)
    tmp_a = np.zeros((num, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((num, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, num):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]  # 第i行存放delay i天的数据
    tmp_a -= np.nanmean(tmp_a, axis=0)
    tmp_b -= np.nanmean(tmp_b, axis=0)
    beta = np.nansum(tmp_a * tmp_b, axis=0) / np.nansum(tmp_a ** 2, axis=0)
    s[num - 1:] = (tmp_b[0] - beta * tmp_a[0])[num - 1:]
    s[:num-1] = np.nan
    return s


# @nb.jit
def tscorr(a: np.array, b: np.array, num: int) -> np.array:  # 日频的时序相关性
    s = np.zeros(a.shape)
    tmp_a = np.zeros((num, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((num, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, num):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]  # 第i行存放delay i天的数据
    tmp_a -= np.nanmean(tmp_a, axis=0)
    tmp_b -= np.nanmean(tmp_b, axis=0)
    s[num - 1:] = (np.nanmean(tmp_a * tmp_b,
                              axis=0) / (np.nanstd(tmp_a, axis=0) * np.nanstd(tmp_b, axis=0)))[num - 1:]
    s[:num - 1] = np.nan
    return s


# @nb.jit
def tscov(a: np.array, b: np.array, num: int) -> np.array:  # 日频的时序协方差
    s = np.zeros(a.shape)
    tmp_a = np.zeros((num, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((num, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, num):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]  # 第i行存放delay i天的数据
    tmp_a -= np.nanmean(tmp_a, axis=0)
    tmp_b -= np.nanmean(tmp_b, axis=0)
    s[num - 1:] = (np.nanmean(tmp_a * tmp_b, axis=0))[num - 1:]
    s[:num - 1] = np.nan
    return s