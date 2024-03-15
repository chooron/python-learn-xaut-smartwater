# coding=utf-8

"""
@author: jingxin
@license: MIT
@contact: jingxin0107@qq.com
@file: demo3.py
@date: 2024/3/15 20:50
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Reservoir:

    # 构造函数
    def __init__(self, name, index):
        self.__name = name
        self.__index = index

        self.init_storage = 0.0

        # 计算的中间状态
        self.storage_list = []
        self.level_list = []

        # 水位插值方法
        self.level_itp = None

    def set_init_storage(self, storage):
        self.init_storage = storage

    @property
    def name(self):
        return self.__name

    @property
    def index(self):
        return self.__index

    # 特征曲线差值
    def level_interp(self, x, y):
        self.level_itp = interp1d(x, y, fill_value='extrapolate')

    # 调节计算
    def calculate(self, inflow, outflow):
        cur_storage = self.init_storage
        for i, o in zip(inflow, outflow):
            tmp_change_storage = (i - o) * 3600 / 1e8
            cur_storage = cur_storage + tmp_change_storage
            cur_level = self.level_itp(cur_storage)
            self.storage_list.append(cur_storage)
            self.level_list.append(cur_level)

    # 输出结果，水库水位
    def get_level_list(self):
        return self.level_list

    # 绘图
    @staticmethod
    def plot_result(level_list):
        # 绘制水库水位过程曲线
        plt.plot(level_list)
        plt.show()


if __name__ == '__main__':
    # 水库基本信息
    name = "黄金峡水库"
    idx = "10001"
    init_storage = 0.694

    # 水库特征曲线信息
    levels = [22, 24, 26, 28, 30, 32]
    storages = [0.61, 0.694, 0.876, 1.133, 1.450, 1.852]

    # 入库流量和出库流量数据
    inflow = np.array([28, 69, 105, 367, 1320, 2440, 2760, 3020,
                       3140, 2900, 2750, 1870, 1300, 1100, 980, 820])
    outflow = np.ones_like(inflow) * 1500

    ss = Reservoir(name, idx)

    # 初始化
    ss.set_init_storage(init_storage)
    ss.level_interp(storages,levels)

    ss.calculate(inflow, outflow)

    level_list2 = ss.get_level_list()
    ss.plot_result(level_list2)
