# coding=utf-8

"""
@author: jingxin
@license: MIT
@contact: jingxin0107@qq.com
@file: demo.py
@date: 2024/3/14 16:27
@desc: 水库调节计算对象化封装练习
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#
#
# storage_list = []
# level_list = []
#
# # 开始调节计算
# cur_storage = init_storage
# for i, o in zip(inflow, outflow):
#     tmp_change_storage = (i - o) * 3600 / 1e8
#     cur_storage = cur_storage + tmp_change_storage
#     cur_level = level_itp(cur_storage)
#     storage_list.append(cur_storage)
#     level_list.append(cur_level)
#
# # 绘制水库水位过程曲线
# plt.plot(level_list)
# plt.show()

class Reservoir:

    # 构造函数
    def __init__(self, ):
        pass

    # 调节计算
    def calculate(self):
        pass

    # 特征曲线插值
    def res_interp(self):
        pass

    # 输出结果，水库水位
    def get_levels(self):
        pass

    # 绘图
    @staticmethod
    def plot_result():
        pass


if __name__ == '__main__':
    # 水库基本信息
    name = "黄金峡水库"
    idx = "10001"
    init_storage = 0.694

    # 水库特征曲线信息
    levels = [22, 24, 26, 28, 30, 32]
    storages = [0.61, 0.694, 0.876, 1.133, 1.450, 1.852]
    # 插值
    storage_itp = interp1d(levels, storages, fill_value='extrapolate')
    level_itp = interp1d(storages, levels, fill_value='extrapolate')

    # 入库流量和出库流量数据
    inflow = np.array([28, 69, 105, 367, 1320, 2440, 2760, 3020,
                       3140, 2900, 2750, 1870, 1300, 1100, 980, 820])
    outflow = np.ones_like(inflow) * 1500

    # 定义一个对象

    # 调用对象方法