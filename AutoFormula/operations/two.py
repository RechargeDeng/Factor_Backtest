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
该代码定义2型运算符
"""


import numpy as np
import numba as nb


def add(a, b):
    return a + b


def minus(a, b):
    return a - b


def prod(a, b):
    c = a * b
    c[np.isnan(c)] = 0
    c[np.isinf(c)] = 0
    return a * b


def div(a, b):
    s = np.zeros(a.shape)
    if type(b) == float:
        s = a / b
    else:
        s[b != 0] = a[b != 0] / b[b != 0]
    return s


# @nb.jit
def intratsregres(a, b):  # 日内时序回归残差
    tmp_a = a.transpose(1, 0, 2)
    tmp_b = b.transpose(1, 0, 2)
    tmp_a -= np.nanmean(tmp_a, axis=0)
    tmp_b -= np.nanmean(tmp_b, axis=0)
    beta = np.nansum(tmp_a * tmp_b, axis=0) / np.nansum(tmp_a ** 2, axis=0)
    s = tmp_b - beta * tmp_a
    return s


def lt(a, b):
    return a < b


def le(a, b):
    return a <= b


def gt(a, b):
    return a > b


def ge(a, b):
    return a >= b