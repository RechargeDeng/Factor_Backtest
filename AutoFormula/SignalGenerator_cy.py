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
SignalGenerator类的Cython版本
"""


import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/High-Frequency-Predictor')
sys.path.append('C:/Users/HBG/Desktop/Repositories/High-Frequency-Predictor')

from AutoFormula.operations_cy.one import *
from AutoFormula.operations_cy.one_num import *
from AutoFormula.operations_cy.one_num_num import *
from AutoFormula.operations_cy.one_num_num_num import *
from AutoFormula.operations_cy.two import *
from AutoFormula.operations_cy.two_num import *
from AutoFormula.operations_cy.two_num_num import *
from AutoFormula.operations_cy.two_num_num_num import *


class SignalGenerator:
    def __init__(self, data):
        """
        :param data: Data类的实例
        """
        self.operation_dic = {}
        self.get_operation()
        self.data = data


        """
        截面算子，因为要调用top
        """

    def csrank(self, a):
        b = a.copy()  # 保持a中的nan
        for i in range(len(a)):
            n = np.sum(~np.isnan(b[i, self.data.top[i]]))
            if n == 0:  # 全是nan就不排序了
                continue
            tmp = b[i, self.data.top[i]].copy()
            valid_tmp = tmp[~np.isnan(tmp)].copy()  # 取出不是nan的部分
            pos = valid_tmp.argsort()
            for j in range(len(pos)):
                valid_tmp[pos[j]] = j
            valid_tmp /= (len(valid_tmp) - 1)
            tmp[~np.isnan(tmp)] = valid_tmp
            b[i, self.data.top[i]] = tmp
        return b

    def zscore(self, a):
        s = a.copy()
        for i in range(len(a)):
            if np.sum(~np.isnan(s[i][self.data.top[i]])) <= 1:
                continue
            s[i][self.data.top[i]] -= np.nanmean(b[i][self.data.top[i]])
            b[i][self.data.top[i]] /= np.nanstd(b[i][self.data.top[i]])
            b[i][(self.data.top[i]) & (b[i] > 3)] = 3
            b[i][(self.data.top[i]) & (b[i] < -3)] = -3
        return b

    def csindneutral_2d(self, a):  # 截面中性化
        for i in range(len(a)):
            se = (self.data.top[i]) & (~np.isnan(a[i]))
            for j in range(-1, self.data.max_ind_code + 1):
                tmp_se = se & (self.data.industry[i] == j)
                if np.sum(tmp_se) >= 1:
                    a[i, tmp_se] -= np.mean(a[i, tmp_se])
        return a

    def csindneutral_3d(self, a):
        for i in range(len(a)):
            se = (self.data.top[i]) & (~np.isnan(a[i]))
            for j in range(-1, self.data.max_ind_code + 1):
                tmp_se = se & (self.data.industry[i] == j)
                if np.sum(tmp_se) >= 1:
                    a[i, tmp_se] -= np.mean(a[i, tmp_se])
        return a

    def csind(self, a):  # 截面替换成所处行业的均值
        s = a.copy()
        ind = self.data.industry['sws']  # 申万二级行业的位置
        for i in range(len(s)):
            ind_num_dic = {}  # 存放行业总数
            ind_sum_dic = {}  # 存放行业总值
            for j in list(set(ind[i])):
                ind_num_dic[j] = np.sum(ind[i] == j)
                ind_sum_dic[j] = np.sum(a[i, ind[i] == j])
            for key in ind_sum_dic.keys():
                ind_sum_dic[key] /= ind_num_dic[key]
            for j in range(s.shape[1]):
                s[i, j] = ind_sum_dic[ind[i, j]]  # 减去行业平均，如果是没有出现过的行业，那么就是0
        return s

    def truncate(self, a, s, e):  # 将过大过小的信号截断为平均值，注意是平均值
        """
        :param a: 数据
        :param s: 起始
        :param e: 结束
        :return: 截断后的信号
        """
        b = self.csrank(a)
        sig = a.copy()
        for i in range(len(a)):
            mean = np.mean(s[i, self.data.top[i]])
            sig[i, self.data.top[i] & ((b[self.data.top[i]] < s) or (b[self.data.top[i]] > e))] = mean
        return sig

    def marketbeta(self, a, ts_window):  # 获得信号和市场平均回望ts_window的beta系数
        s = np.zeros(a.shape)
        if ts_window < 2:
            ts_window = 2  # 至少回望两天
        mar_mean = np.zeros(len(a))
        for i in range(len(a)):
            if np.sum(~np.isnan(a[i, self.data.top[i]])) > 0:
                mar_mean[i] = np.nanmean(a[i, self.data.top[i]])
            else:
                mar_mean[i] = np.nan
        tmp_a = np.zeros(ts_window, a.shape[1], a.shape[0])  # 必须是ts_window * cs * ts
        tmp_a[0] = a.copy().T
        for i in range(1, num):
            tmp_a[i, :, i:] = a[:-i].T  # 第i列存放delay i天的数据
        tmp_m = np.zeros((num, a.shape[1], a.shape[0]))
        tmp_m[0] = mar_mean
        for i in range(1, num):
            tmp_m[i, :, i:] = mar_mean[:-i]  # 第i列存放delay i天的数据
        tmp_a = tmp_a.transpose(0, 2, 1)
        tmp_m = tmp_m.transpose(0, 2, 1)
        tmp_a -= np.nanmean(tmp_a, axis=0)
        tmp_m -= np.nanmean(tmp_m, axis=0)
        s[num - 1:] = (np.nanmean(tmp_a * tmp_m,
                                  axis=0) / (np.nanstd(tmp_b, axis=0)))[num - 1:]
        s[:num - 1] = np.nan
        return s

    def discrete(self, a, num):  # 离散化算子，将截面信号离散成0到num-1的整数
        b = a.copy()  # 复制主要是保持a中本来是nan的部分也为nan
        for i in range(len(a)):
            n = np.sum(~np.isnan(b[i, self.data.top[i]]))
            if n == 0:  # 说明全是nan
                continue
            tmp = b[i, self.data.top[i]].copy()
            valid_tmp = tmp[~np.isnan(tmp)].copy()  # 取出不是nan的部分
            pos = valid_tmp.argsort()
            for j in range(num-1):
                se = (j * (len(pos) // num) <= pos) & (pos < (j + 1) * (len(pos) // num))
                valid_tmp[pos[se]] = j
            se = (num - 1) * (len(pos) // num) <= pos
            valid_tmp[pos[se]] = num - 1
            tmp[~np.isnan(tmp)] = valid_tmp  # 排序后再赋值回来
            b[i, self.data.top[i]] = tmp
        return b

    def get_operation(self):

        # 1型算符
        self.operation_dic['neg_2d'] = neg_2d
        self.operation_dic['neg_3d'] = neg_3d
        self.operation_dic['absv_2d'] = absv_2d
        self.operation_dic['absv_3d'] = absv_3d
        self.operation_dic['log_2d'] = log_2d
        self.operation_dic['log_3d'] = log_3d
        self.operation_dic['logv_2d'] = logv_2d
        self.operation_dic['logv_3d'] = logv_3d
        self.operation_dic['intratsfftreal'] = intratsfftreal
        self.operation_dic['intratsfftimag'] = intratsfftimag

        # 1_num型运算符
        self.operation_dic['powv_2d'] = powv_2d
        self.operation_dic['powv_3d'] = powv_3d
        self.operation_dic['tsmax_2d'] = tsmax_2d
        self.operation_dic['tsmax_3d'] = tsmax_3d
        self.operation_dic['intratsmax_3d'] = intratsmax_3d
        self.operation_dic['tsmaxpos_2d'] = tsmaxpos_2d
        self.operation_dic['tsmaxpos_3d'] = tsmaxpos_3d
        self.operation_dic['tsmin_2d'] = tsmin_2d
        self.operation_dic['tsmin_3d'] = tsmin_3d
        self.operation_dic['tsminpos_2d'] = tsminpos_2d
        self.operation_dic['tsminpos_3d'] = tsminpos_3d
        self.operation_dic['tsdelay_2d'] = tsdelay_2d
        self.operation_dic['tsdelay_3d'] = tsdelay_3d
        self.operation_dic['tsdelta_2d'] = tsdelta_2d
        self.operation_dic['tsdelta_3d'] = tsdelta_3d
        self.operation_dic['tspct_2d'] = tspct_2d
        self.operation_dic['tspct_3d'] = tspct_3d
        self.operation_dic['tsstd_2d'] = tsstd_2d
        self.operation_dic['tsstd_3d'] = tsstd_3d
        self.operation_dic['tsmean_2d'] = tsmean_2d
        self.operation_dic['tsmean_3d'] = tsmean_3d
        self.operation_dic['tskurtosis_2d'] = tskurtosis_2d
        self.operation_dic['tskurtosis_3d'] = tskurtosis_3d
        self.operation_dic['tsskew_2d'] = tsskew_2d
        self.operation_dic['tsskew_3d'] = tsskew_3d
        self.operation_dic['wdirect'] = wdirect
        self.operation_dic['tsrank_2d'] = tsrank_2d
        self.operation_dic['tsrank_3d'] = tsrank_3d
        self.operation_dic['intratshpf'] = intratshpf
        self.operation_dic['intratslpf'] = intratslpf

        # 1_num_num型算子
        self.operation_dic['intratsmax_3d'] = intratsmax_3d
        self.operation_dic['intratsmaxpos_3d'] = intratsmaxpos_3d
        self.operation_dic['intratsmin_3d'] = intratsmin_3d
        self.operation_dic['intratsminpos_3d'] = intratsminpos_3d
        self.operation_dic['intratsstd_3d'] = intratsstd_3d
        self.operation_dic['intratsmean_3d'] = intratsmean_3d
        self.operation_dic['intratskurtosis_3d'] = intratskurtosis_3d
        self.operation_dic['intratsskew_3d'] = intratsskew_3d
        # self.operation_dic['tsautocorr'] = tsautocorr
        self.operation_dic['tsfftreal'] = tsfftreal
        self.operation_dic['tsfftimag'] = tsfftimag
        self.operation_dic['tshpf'] = tshpf
        self.operation_dic['tslpf'] = tslpf
        self.operation_dic['tsquantile_2d'] = tsquantile_2d
        self.operation_dic['tsquantile_3d'] = tsquantile_3d
        self.operation_dic['tsquantileupmean_2d'] = tsquantileupmean_2d
        self.operation_dic['tsquantileupmean_3d'] = tsquantileupmean_3d
        self.operation_dic['tsquantiledownmean_2d'] = tsquantiledownmean_2d
        self.operation_dic['tsquantiledownmean_3d'] = tsquantiledownmean_3d

        # 1_num_num_num型算符
        self.operation_dic['intraquantile_3d'] = intraquantile_3d
        self.operation_dic['intraquantileupmean_3d'] = intraquantileupmean_3d
        self.operation_dic['intraquantiledownmean_3d'] = intraquantiledownmean_3d

        # 2型运算符
        self.operation_dic['add_2d'] = add_2d
        self.operation_dic['add_num_2d'] = add_num_2d
        self.operation_dic['add_3d'] = add_3d
        self.operation_dic['add_num_3d'] = add_num_3d
        self.operation_dic['minus_2d'] = minus_2d
        self.operation_dic['minus_num_2d'] = minus_num_2d
        self.operation_dic['minus_3d'] = minus_3d
        self.operation_dic['minus_num_3d'] = minus_num_3d
        self.operation_dic['prod_2d'] = prod_2d
        self.operation_dic['prod_num_2d'] = prod_num_2d
        self.operation_dic['prod_3d'] = prod_3d
        self.operation_dic['prod_num_3d'] = prod_num_3d
        self.operation_dic['div_2d'] = div_2d
        self.operation_dic['div_num_2d'] = div_num_2d
        self.operation_dic['div_3d'] = div_3d
        self.operation_dic['div_num_3d'] = div_num_3d
        self.operation_dic['intratsregres_3d'] = intratsregres_3d
        self.operation_dic['lt_2d'] = lt_2d
        self.operation_dic['lt_3d'] = lt_3d
        self.operation_dic['le_2d'] = le_2d
        self.operation_dic['le_3d'] = le_3d
        self.operation_dic['gt_2d'] = gt_2d
        self.operation_dic['gt_3d'] = gt_3d
        self.operation_dic['ge_2d'] = ge_2d
        self.operation_dic['ge_3d'] = ge_3d

        # 2_num型运算符
        self.operation_dic['tsregres_2d'] = tsregres_2d
        self.operation_dic['tsregres_3d'] = tsregres_3d
        self.operation_dic['tscorr_2d'] = tscorr_2d
        self.operation_dic['tscorr_3d'] = tscorr_3d

        # 2_num_num型运算符
        self.operation_dic['bitsquantile_2d'] = bitsquantile_2d
        self.operation_dic['bitsquantile_3d'] = bitsquantile_3d
        self.operation_dic['bitsquantileupmean_2d'] = bitsquantileupmean_2d
        self.operation_dic['bitsquantileupmean_3d'] = bitsquantileupmean_3d
        self.operation_dic['bitsquantiledownmean_2d'] = bitsquantiledownmean_2d
        self.operation_dic['bitsquantiledownmean_3d'] = bitsquantiledownmean_3d

        # 2_num_num_num型运算符
        self.operation_dic['tssubset_3d'] = tssubset_3d
        self.operation_dic['tssubset_3d'] = tssubset_3d
        self.operation_dic['biintraquantile_3d'] = biintraquantile_3d
        self.operation_dic['biintraquantileupmean_3d'] = biintraquantileupmean_3d
        self.operation_dic['biintraquantiledownmean_3d'] = biintraquantiledownmean_3d

        def condition(a, b, c):
            """
            :param a: 条件，一个布尔型矩阵
            :param b: 真的取值
            :param c: 假的取值
            :return: 信号
            """
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if type(b) == int or type(b) == float:
                    s[i, a[i]] = b
                else:
                    s[i, a[i]] = b[i, a[i]]
                if type(c) == int or type(c) == float:
                    s[i, ~a[i]] = c
                else:
                    s[i, ~a[i]] = c[i, ~a[i]]
            return s

        self.operation_dic['condition'] = condition
