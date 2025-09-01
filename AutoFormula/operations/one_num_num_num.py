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
该代码定义1_num_num_num型运算符
"""


import numpy as np
import numba as nb


def intraquantile(a, start, end, num):  # 日内分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入，返回2维矩阵
    tmp = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp = np.sort(tmp, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))
    s = tmp[pos]
    return s


def intraquantileupmean(a, start, end, num):  # 日内分位数上行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    tmp = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp = np.sort(tmp, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))
    s = np.mean(tmp[pos:], axis=0)
    return s


def intraquantiledownmean(a, start, end, num):  # 日内分位数下行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    tmp = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp = np.sort(tmp, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))
    s = np.mean(tmp[:pos + 1], axis=0)
    return s
