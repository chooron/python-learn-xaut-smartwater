# coding=utf-8

"""
@author: jingxin
@license: MIT
@contact: jingxin0107@qq.com
@file: demo2.py
@date: 2024/3/15 20:33
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#
# 水库基本信息
name = "黄金峡水库"
idx = "10001"
init_storage = 0.694
storage_list = []
level_list = []
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

# 开始调节计算
cur_storage = init_storage
for i, o in zip(inflow, outflow):
    tmp_change_storage = (i - o) * 3600 / 1e8
    cur_storage = cur_storage + tmp_change_storage
    cur_level = level_itp(cur_storage)
    storage_list.append(cur_storage)
    level_list.append(cur_level)

# 绘制水库水位过程曲线
plt.plot(level_list)
plt.show()