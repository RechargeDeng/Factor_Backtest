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
该代码定义1型运算符
"""


import numpy as np
import numba as nb


def neg(a):
    return -a


def absv(a):  # 取绝对值
    return np.abs(a)


def intratsfftreal(a):  # 日内fft实数部分
    return np.fft.fft(a, axis=1).real / a.shape[1]  # 归一化


def intratsfftimag(a):  # 日内fft虚数部分
    return np.fft.fft(a, axis=1).imag / a.shape[1]  # 归一化
