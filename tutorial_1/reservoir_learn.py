# coding=utf-8

"""
@author: jingxin
@license: MIT
@contact: jingxin0107@qq.com
@file: reservoir_learn.py
@date: 2024/3/8 20:59
@desc: 
"""
# 定义水库对象
import datetime

import numpy as np
from scipy.interpolate import interp1d
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination


class Curve:
    def __init__(self, levels, storages):
        self.levels = levels
        self.storages = storages
        self.interps = {'storage': interp1d(levels, storages, fill_value='extrapolate'),
                        'level': interp1d(storages, levels, fill_value='extrapolate')}

    def transform(self, value, interp_type):
        return self.interps.get(interp_type)(value)


class Reservoir:

    def __init__(self, id, name, curve, start_state, **kwargs):
        """
        基本水库对象(描述水库运行原理)
        :param id: 水库id
        :param name: 水库名称
        :param curve: 水库水位、库容曲线
        :param kwargs: 水库的一些其他特性:
                        1. 水库群拓扑关系
                        2. 调度周期 dict: {'day|month': num}
                        ...由于不是专业干水库调度, 故了解不够
        """
        # 静态变量
        self.id = id
        self.name = name
        self.curve = curve
        if 'operation_period' not in kwargs.keys():
            self.operation_period = datetime.timedelta(**{'hours': 1})
        else:
            self.operation_period = datetime.timedelta(**kwargs['operation_period'])

        # 动态变量
        self.inflow_list = []
        self.outflow_list = []
        self.storage_list = []
        self.level_list = []
        self.state = self.start_state = start_state

    @classmethod
    def build_by_proto(cls, proto_obj, start_state):
        curve = Curve(list(proto_obj.curve.levels), list(proto_obj.curve.storages))
        id = proto_obj.id
        name = proto_obj.name
        return cls(id, name, curve, start_state)

    def run(self, inflow, outflow):
        """
        基于水量平衡方程的离散化调度优化对象为出库流量
        :param inflow: 水库入流量(m^3/s)
        :param outflow: 水库出流量(m^3/s)
        :param start_state: 计算初水库水位(m)
        :return: None
        """
        # 以日调度为例计算入流出流后水库的水量变化
        # note: 这里的计算方法稍微有差异，有的是计算时段初末的平均值
        avg_inflow = (inflow + self.state['inflow']) / 2
        avg_outflow = (outflow + self.state['outflow']) / 2
        end_storage = self.state['storage'] + (avg_inflow - avg_outflow) * self.operation_period.total_seconds()
        end_level = self.curve.transform(end_storage, 'level').item()
        self.update(inflow, outflow, end_storage, end_level)

    def runv2(self, inflow, end_level):
        """
        基于水量平衡方程的离散化调度优化对象为水位
        :param inflow: 水库入流量(m^3/s)
        :param end_level: 水库末水位(m^3/s)
        :return: None
        """
        # 以日调度为例计算入流出流后水库的水量变化
        # note: 这里的计算方法稍微有差异，有的是计算时段初末的平均值
        avg_inflow = (inflow + self.start_state['inflow']) / 2
        end_storage = self.curve.transform(end_level, 'storage').item()
        change_storage = end_storage - self.start_state['storage']
        avg_outflow = avg_inflow - change_storage / self.operation_period.total_seconds()
        outflow = avg_outflow * 2 - self.start_state['outflow']
        self.update(inflow, outflow, end_storage, end_level)

    def update(self, inflow, outflow, storage, level):
        """
        更新水库动态参数
        :param inflow: 实时入流量
        :param outflow: 实时出流量
        :param storage: 实时库容
        :param level: 实时水位
        :return: None
        """
        self.inflow_list.append(inflow)
        self.outflow_list.append(outflow)
        self.storage_list.append(storage)
        self.level_list.append(level)
        self.state = {'inflow': inflow, 'outflow': outflow, 'storage': storage, 'level': level}

    def clear(self):
        self.inflow_list = []
        self.outflow_list = []
        self.storage_list = []
        self.level_list = []
        self.state = self.start_state


class SimpleFloodOperation(ElementwiseProblem):

    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu,
                 reservoir=None, inflow_series=None, start_state=None,
                 max_level=25, min_level=21, max_sum_outflow=1):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.reservoir = reservoir
        self.inflow_series = inflow_series
        self.start_state = start_state
        self.max_level = max_level
        self.min_level = min_level
        self.max_sum_outflow = max_sum_outflow

    def _evaluate(self, x, out, *args, **kwargs):
        self.reservoir.clear()
        for inflow, outflow in zip(self.inflow_series, x):
            self.reservoir.run(inflow, outflow)
        max_level = max(self.reservoir.level_list)
        min_level = min(self.reservoir.level_list)
        max_outflow = max(self.reservoir.outflow_list)

        level_limit1 = max_level - self.max_level
        level_limit2 = self.min_level - min_level

        out['F'] = [max_outflow, max_level]
        out['G'] = [level_limit1, level_limit2]
        self.reservoir.clear()


if __name__ == '__main__':
    algorithm = NSGA2(
        pop_size=50,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20, prob_var=0.2),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", 100)

    from pymoo.optimize import minimize

    curve = Curve(levels=[22, 24, 26, 28, 30, 32],
                  storages=[0.61e8, 0.694e8, 0.876e8, 1.133e8, 1.450e8, 1.852e8])
    start_storage = 0.694e8
    start_state = {'inflow': 0, 'outflow': 0, 'storage': start_storage}
    reservoir_obj = Reservoir(id=10001, name='test', curve=curve,
                              start_state=start_state)

    inflow = [28, 69, 105, 367, 1320, 2440, 2760, 3020,
              3140, 2900, 2750, 1870, 1300, 1100, 980, 820]
    xl = np.array([0] * len(inflow))
    xu = np.array([5e3] * len(inflow))
    problem = SimpleFloodOperation(n_var=len(inflow), n_obj=2, n_ieq_constr=2, xl=xl, xu=xu,
                                   reservoir=reservoir_obj, inflow_series=inflow,
                                   max_level=30, min_level=20)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F
    fl = res.F.argmin(axis=0)[0]
    best_outflow = res.X[fl, :].tolist()
    reservoir_objv2 = Reservoir(id=10001, name='test', curve=curve,
                                start_state=start_state)
    for inflow, outflow in zip(inflow, best_outflow):
        reservoir_objv2.run(inflow, outflow)
