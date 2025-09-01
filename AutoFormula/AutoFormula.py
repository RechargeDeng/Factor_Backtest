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
AutoFormula
该模块定义调用FormulaTree进行解析公式以及测试
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/High-Frequency-Predictor')
from DataLoader.DataLoader import Data
from AutoFormula.AutoTester import AutoTester, Stats
from AutoFormula.FormulaTree import FormulaTree, Node, FormulaParser
from SignalGenerator import SignalGenerator


class AutoFormula:
    def __init__(self, data: Data):
        """
        :param data: Data实例
        """
        self.tree_generator = FormulaTree()
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()
        self.auto_tester = AutoTester()

    def cal_formula(self, tree: Node, data_dic: dict, return_type: str = 'signal') -> np.array:  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param data_dic: 原始数据的字典，可以通过字段读取对应的矩阵
        :param return_type: 返回值形式
        :return: 返回计算好的signal矩阵
        """
        if return_type == 'signal':
            #print("变量类型:", tree.variable_type)
            # print("操作类型:", tree.operation_type)
            # print("名称:", tree.name)
            if tree.variable_type == 'data':
                if type(tree.name) == int or type(tree.name) == float:
                    return tree.name  # 直接挂载在节点上，但是应该修改成需要数字的就直接返回数字
                #print(data_dic[tree.name].copy())
                return data_dic[tree.name].copy()  # 当前版本需要返回一个副本
            elif tree.variable_type == 'intra_data':
                if tree.num_1 is not None:
                    return data_dic[tree.name][:, tree.num_1, :].copy()
                else:
                    return data_dic[tree.name].copy()  # 如果没有数字就直接返回原本的数据
            else:
                if tree.operation_type == '1':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type))
                if tree.operation_type == '1_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1)
                if tree.operation_type == '1_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1, tree.num_2)
                if tree.operation_type == '1_num_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1, tree.num_2, tree.num_3)
                if tree.operation_type == '2':  # 此时需要判断有没有数字
                    if tree.num_1 is None:
                        return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                                                                                        return_type),
                                                                       self.cal_formula(tree.right, data_dic,
                                                                                        return_type))
                    else:
                        if tree.left is not None:
                            return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                                                                                            return_type),
                                                                           tree.num_1)
                        else:
                            return self.operation.operation_dic[tree.name](tree.num_1,
                                                                           self.cal_formula(tree.right, data_dic,
                                                                                            return_type))
                if tree.operation_type == '2_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1)
                if tree.operation_type == '2_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1, tree.num_2)
                if tree.operation_type == '2_num_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1, tree.num_2, tree.num_3)
                if tree.operation_type == '3':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.middle, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type))
        if return_type == 'str':
            if tree.variable_type == 'data':
                return tree.name  # 返回字符串
            elif tree.variable_type == 'intra_data':  # 这里也需要判断是否有数字
                if tree.num_1 is not None:
                    return '{' + tree.name + ',{}'.format(tree.num_1) + '}'
                else:
                    return '{' + tree.name + '}'
            else:
                if tree.operation_type == '1':
                    return tree.name + '{' + (self.cal_formula(tree.left, data_dic, return_type)) + '}'
                if tree.operation_type == '1_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + '}'
                if tree.operation_type == '1_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '1_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '2':  # 此时需要判断是否有数字
                    if tree.num_1 is not None:
                        return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                               self.cal_formula(tree.right, data_dic, return_type) + '}'
                    else:
                        if tree.left is not None:
                            return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                                   str(tree.num_1) + '}'
                        else:
                            return tree.name + '{' + str(tree.num_1) + ',' + \
                                   self.cal_formula(tree.right, data_dic, return_type) + '}'
                if tree.operation_type == '2_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + '}'
                if tree.operation_type == '2_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '2_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '3':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.middle, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + '}'

    def test_formula(self, formula: str, data: Data, start: int = 100,
                     end: int = 4600, shift: int = 1) -> (Stats, np.array):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param data: Data类
        :param: start: 每日测试开始的snap
        :param: end: 每日测试结束的snap
        :param: shift: 预测平移量
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        # print(formula.operation_type)
        # print(formula.left)
        # print(formula.left.left)
        signal = self.cal_formula(formula, data.data_dic)  # 暂时为了方便，无论如何都计算整个回测区间的因子值
        time_matrix = data.data_dic['close_time']
        time_matrix_1d = time_matrix.flatten(order='F')
        signal_1d = signal.flatten(order='F')
        signal_df = pd.DataFrame(
            {'time': time_matrix_1d,
            'signal': signal_1d}
        )
        # print(signal)
        # print(signal.shape)
        # print(signal_df)
        # print(signal_df.shape)
        # print(signal_1d.shape)
        
        #signal_df.to_csv('signal_df.csv', index=False)
       #print(signal.shape)
        return self.auto_tester.test(signal, data.ret, start=start, end=end, shift=shift), signal, signal_df
    def generate_signal(self, formula: str, data: Data):
        """
        it works for living signal generation
        """
        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        signal = self.cal_formula(formula, data.data_dic)
        return signal