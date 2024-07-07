# -*- coding:utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from cvxpy import Variable, Minimize, sum
from scipy.io import loadmat
import matplotlib.patches as patches


def cvxSchedule(evtmp, Tcur):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 构建.mat文件的绝对路径
    datas545_path = os.path.join(script_dir, "data", "datas545.mat")
    evfile_path = os.path.join(script_dir, "data", "evfile545.mat")
    sensitivity_y_path = os.path.join(script_dir, "data", "sensitivity_y.mat")

    # 尝试加载.mat文件
    try:
        datas545 = loadmat(datas545_path)
        evfile = loadmat(evfile_path)
        sensitivity_y = loadmat(sensitivity_y_path)
    except FileNotFoundError:
        print("Error: One or more files not found.")
    evtmp = evtmp[:, 1:]  # 删除第一列
    _, EVnum = evtmp.shape
    print("可调度车辆", EVnum)
    a = 1
    b = 1
    c = 0.00001
    d = 0.01
    Pchar = 7  # 最大充电功率
    Pdis = -7  # 最大放电功率
    Eini = evtmp[0, :]  # 初始soc
    Efin = evtmp[1, :]  # 目标电量
    Ecap = np.ones(EVnum) * 53.1  # 全部初始化为53.1
    Ezero = np.zeros(EVnum)
    Tleft = (evtmp[3, :] - Tcur).astype(int)  # 剩余调度时间
    time = int(max(Tleft))
    # 检查是否结束
    if time < 1:
        schedule = np.zeros(EVnum, 1)
        return schedule, 0
    Tcur = int(Tcur)
    print("调度开始时间：%d,可调度时间点：%d", Tcur, time)

    baseload = datas545["dayload"][Tcur : Tcur + time]
    baseload = np.squeeze(baseload)  # 二维转一维
    sense = sensitivity_y["sensitivity"][:, Tcur : Tcur + time]
    # 凸优化
    z: Variable = Variable(time)
    x: Variable = Variable((EVnum, time))
    fitness: Variable = Variable(time)
    # batloss: Variable = Variable(time)
    objective = cp.Minimize(
        cp.sum(a * z + 0.5 * b * z**2 - a * baseload - 0.5 * b * baseload**2 + fitness)
    )
    # 约束条件
    # 定义约束条件
    constraints = [
        z
        == baseload + cp.sum(x, axis=0)
        # fitness == cp.sum(cp.multiply(sense[:, :time], x), axis=0),
    ]
    for i in range(time):
        subfit = 0
        for j in range(EVnum):
            node = int(evtmp[5, j])
            subfit += sense[node, i] * x[j, i]
        constraints.append(fitness[i] == subfit)
    # 离网时间限制：可调度范围之外的充放电功率均为0
    for i in range(EVnum):
        if Tleft[i] < time:
            constraints.append(cp.constraints.Zero(x[i, Tleft[i] : time]))
    # 充放电速率限制
    for i in range(EVnum):
        if evtmp[4, i] == 1:
            constraints += [Pdis <= x[i, :], x[i, :] <= Pchar]
        else:
            constraints.append(0 <= x[i, :] <= Pchar)
    # 每时刻的电量限制
    for i in range(time):
        constraints += [
            Ezero <= Eini + cp.sum(x[:, : i + 1], axis=1),
            Eini + cp.sum(x[:, : i + 1], axis=1) <= Ecap,
        ]
    # 离网时总电流大于目标电量
    constraints.append(Eini + cp.sum(x, axis=1) >= Efin)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    schedule = x[:, 0].value
    optval = problem.value
    return schedule, optval


def function_for_Community():
    # 加载数据
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 构建.mat文件的绝对路径
    datas545_path = os.path.join(script_dir, "datas545.mat")
    evfile_path = os.path.join(script_dir, "evfile545.mat")
    sensitivity_y_path = os.path.join(script_dir, "sensitivity_y.mat")

    # 尝试加载.mat文件
    try:
        datas545 = loadmat(datas545_path)
        evfile = loadmat(evfile_path)
        sensitivity_y = loadmat(sensitivity_y_path)
    except FileNotFoundError:
        print("Error: One or more files not found.")
    evfile545 = evfile["evfile"]
    print("6666")
    # 参数设置
    time = 96
    tau = 0.25
    EVnum = datas545["dayev"].sum()
    scheLine = np.zeros((EVnum, time))  # 时间矩阵
    socLine = np.zeros((EVnum, time + 1))  # 电量矩阵
    socLine[:, 0] = evfile545[0, :]  # 写入初始电量
    optvalLine = np.zeros(time)
    for Tcur in range(time):
        point = datas545["dayev"][: Tcur + 1].sum()  # t时刻共接入的电动汽车数
        evtmp = np.zeros((6, 1))  # 单辆EV出力矩阵

        for i in range(point):  # 选择EV充放电策略
            if evfile545[3, i] >= Tcur:  # 离网时间大于当前时刻
                evtmp = np.concatenate(
                    (evtmp, evfile545[:, i].reshape(-1, 1)), axis=1
                )  # 数据更新
        schedule, optval = cvxSchedule(evtmp, Tcur)

        optvalLine[Tcur] = optval  # 将优化值存放到optvalLine当前时刻位置
        if schedule is not None:
            print(f"Optimal Schedule: {schedule}")
            # print(f"Optimal Value: {optval}")
            point2 = 0
            for i in range(point - 1):
                if evfile545[3, i] >= Tcur:  # 离网时间大于当前时刻，定位到调度的车辆
                    evfile545[0, i] += schedule[point2]  # 更新电量
                    scheLine[i, Tcur] = schedule[point2]
                    point2 += 1
                    # print(i, evfile545[0, i])
        else:
            print("Failed to obtain optimal schedule.")
        socLine[:, Tcur + 1] = evfile545[0, :]  # 记录电动汽车电量随时间的变化
        # 计算当前迭代的进度百分比
        progress_percentage = int((Tcur + 1) / time * 100)
        self.progress_updated.emit(progress_percentage)
        # self.scheLine = scheLine
    dayload = np.squeeze(datas545["dayload"])
    z = dayload + np.sum(scheLine, axis=0)
    # 在 function_for_Community 中直接调用 EVgante 函数
    return dayload, z, scheLine, sensitivity_y, datas545


def EVgante(x):
    EVnum, time = x.shape
    plt.axis([0, time, 0, EVnum])
    plt.xlabel("日内时刻")
    plt.ylabel("电动汽车")
    plt.title("电动汽车充放电记录  绿色充电，红色放电")

    colors = ["g", "r"]
    rec = [0, 0, 0, 0]

    for i in range(EVnum):
        for j in range(time):
            if x[i, j] != 0:
                rec[0] = j - 1
                rec[1] = i - 1
                rec[2] = 1
                rec[3] = 1
                color_select = colors[0] if x[i, j] > 0 else colors[1]
                rectangle = patches.Rectangle(
                    (rec[0], rec[1]),
                    rec[2],
                    rec[3],
                    linewidth=0.1,
                    linestyle="-",
                    facecolor=color_select,
                    edgecolor="black",
                )
                plt.gca().add_patch(rectangle)
    plt.show()
