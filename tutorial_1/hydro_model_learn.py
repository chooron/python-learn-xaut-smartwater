# coding=utf-8

"""
@author: jingxin
@license: MIT
@contact: jingxin0107@qq.com
@file: hydro_model_learn.py
@date: 2024/3/8 20:51
@desc: 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


step_fct = lambda x: (np.tanh(5.0 * x) + 1.0) * 0.5
Ps = lambda P, T, Tmin: step_fct(Tmin - T) * P

Pr = lambda P, T, Tmin: step_fct(T - Tmin) * P
M = lambda S0, T, Df, Tmax: (step_fct(T - Tmax) *
                             step_fct(S0) * np.minimum(S0, Df * (T - Tmax)))

PET = lambda T, Lday: (29.8 * Lday * 0.611 *
                       np.exp((17.3 * T) / (T + 237.3)) / (T + 273.2))

ET = lambda S1, T, Lday, Smax: \
    step_fct(S1) * step_fct(S1 - Smax) * PET(T, Lday) + \
    step_fct(S1) * step_fct(Smax - S1) * PET(T, Lday) * (S1 / Smax)

Qb = lambda S1, f, Smax, Qmax: (step_fct(S1) * step_fct(S1 - Smax) * Qmax +
                                step_fct(S1) * step_fct(Smax - S1) * Qmax *
                                np.exp(-f * (Smax - S1)))
Qs = lambda S1, Smax: step_fct(S1) * step_fct(S1 - Smax) * (S1 - Smax)


def exp_hydro_ode(t, S, *args):
    """
    ordinary differential equations of the model for step time point solving
    :param t: time idx
    :param S: stores value
    :param args: model param (Dict)
    :return: dS1, dS2
    """
    S1, S2 = S
    f, Smax, Qmax, Df, Tmax, Tmin, prcp, temp, lday = args

    prcp_, temp_, lday_ = prcp(t), temp(t), lday(t)
    Q_out = Qb(S2, f, Smax, Qmax) + Qs(S2, Smax)
    dS1 = Ps(prcp_, temp_, Tmin) - M(S1, temp_, Df, Tmax)
    dS2 = Pr(prcp_, temp_, Tmin) + M(S1, temp_, Df, Tmax) - ET(S2, temp_, lday_, Smax) - Q_out
    return [dS1, dS2]


# import data
data_path = r'../data/01013500.csv'
data = pd.read_csv(data_path).iloc[:10000, :]

# data interpolate
prcp_itp = interp1d(range(len(data)), data['prcp(mm/day)'].values, fill_value="extrapolate")
tmean_itp = interp1d(range(len(data)), data['tmean(C)'].values, fill_value="extrapolate")
dayl_itp = interp1d(range(len(data)), data['dayl(day)'].values, fill_value="extrapolate")

# set model params
f, Smax, Qmax, Df, Tmax, Tmin = param = (0.01674478, 1709.461015, 18.46996175, 2.674548848, 0.175739196, -2.092959084)

# build ode solver
import time
st_time = time.time()
sol = solve_ivp(exp_hydro_ode,
                [0, len(data)], [0, 1303.004248],
                t_eval=range(len(data)),
                args=param + (prcp_itp, tmean_itp, dayl_itp))

S_snow_series = sol.y[0, :]
S_water_series = sol.y[1, :]

Qb_series = Qb(S_water_series, f, Smax, Qmax)
Qs_series = Qs(S_water_series, Smax)
Q_series = Qb_series + Qs_series

print(time.time()-st_time)

plt.plot(Q_series)
plt.plot(data["flow(mm)"].values[:10000])
plt.show()