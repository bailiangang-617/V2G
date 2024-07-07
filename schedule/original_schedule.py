# -*- coding:utf-8 -*-
import threading
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from cvxpy import Variable
from scipy.io import loadmat
import matplotlib.patches as patches

# import WUXUandRESULT_copy
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy import Variable
from scipy.io import loadmat
import matplotlib.patches as patches

# 生活区--原始数据
SHdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
SHevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
SHsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
SHdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\SHdayload.mat")


def cvxSchedule(evtmp, Tcur, dayload, sensitivity_y):
    global global_thread_id

    # 数据加载
    # datas545 = loadmat(r'C:\Users\Administrator\Desktop\datas545.mat')
    # sensitivity_y = loadmat(r'C:\Users\Administrator\Desktop\sensitivity_y.mat')
    evtmp = evtmp[:, 1:]  # 删除第一列
    _, EVnum = evtmp.shape
    # print('可调度车辆', EVnum)
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

    baseload = dayload[Tcur : Tcur + time]
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
    problem.solve(solver=cp.SCS)
    schedule = x[:, 0].value
    optval = problem.value
    return schedule, optval


def EVgante(x):
    EVnum, time = x.shape
    plt.figure()
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
    plt.savefig(
        r"C:\V2G\Schedule\result\开源数据调度\Charge_and_discharge_records.png"
    )
    plt.show()


def scheduleMain(dayload, datas, evfile, sensitivity_y):
    # 参数设置
    time = 96
    tau = 0.25
    evfile545 = evfile["evfile"]
    EVnum = datas["dayev"].sum()
    scheLine = np.zeros((EVnum, time))  # 时间矩阵
    socLine = np.zeros((EVnum, time + 1))  # 电量矩阵
    socLine[:, 0] = evfile545[0, :]  # 写入初始电量
    optvalLine = np.zeros(time)
    for Tcur in range(time):
        point = datas["dayev"][: Tcur + 1].sum()  # t时刻共接入的电动汽车数
        evtmp = np.zeros((6, 1))  # 单辆EV出力矩阵

        for i in range(point):  # 选择EV充放电策略

            if evfile545[3, i] >= Tcur:  # 离网时间大于当前时刻
                evtmp = np.concatenate(
                    (evtmp, evfile545[:, i].reshape(-1, 1)), axis=1
                )  # 数据更新
        schedule, optval = cvxSchedule(evtmp, Tcur, dayload, sensitivity_y)
        optvalLine[Tcur] = optval  # 将优化值存放到optvalLine当前时刻位置
        if schedule is not None:

            point2 = 0
            for i in range(point - 1):
                if evfile545[3, i] >= Tcur:  # 离网时间大于当前时刻，定位到调度的车辆
                    evfile545[0, i] += schedule[point2]  # 更新电量
                    scheLine[i, Tcur] = schedule[point2]
                    point2 += 1
                    # print(i, evfile545[0, i])
        else:
            print(f" Failed to obtain optimal schedule.")
        socLine[:, Tcur + 1] = evfile545[0, :]  # 记录电动汽车电量随时间的变化

    dayload = np.squeeze(dayload)
    z = dayload + np.sum(scheLine, axis=0)

    #结果保存
    df_z= pd.DataFrame(z)
    df_z.to_csv(r'C:\V2G\Schedule\result\开源数据调度\output_z.csv',index=False,header=False)

    df_socLine= pd.DataFrame(socLine)
    df_socLine.to_csv(r'C:\V2G\Schedule\result\开源数据调度\output_socLine.csv',index=False,header=False)
    
    df_scheLine= pd.DataFrame(scheLine)
    df_scheLine.to_csv(r'C:\V2G\Schedule\result\开源数据调度\output_scheLine.csv',index=False,header=False)

    #结果可视化
    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 使用中文字体
    plt.figure()
    plt.plot(dayload, "r")
    plt.ylabel("负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷"])
    plt.savefig(
        r"C:\V2G\Schedule\result\开源数据调度\Baseload.png"
    )

    # 优化曲线绘制
    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 使用中文字体
    plt.figure()
    plt.plot(dayload, "r")
    plt.plot(z, "b")
    plt.ylabel("负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "优化法接入EV后负荷"])
    plt.savefig(
        r"C:\V2G\Schedule\result\开源数据调度\optimization.png"
    )
    #总充放电量
    plt.figure()
    plt.bar(np.arange(time), np.sum(scheLine, axis=0), edgecolor="black")
    plt.ylabel("EV充放电负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "EV在电网中充放电量"])
    plt.savefig(
        r"C:\V2G\Schedule\result\开源数据调度\EVload.png"
    )
    #充放电记录
    EVgante(scheLine)
    #plt.show()


if __name__ == "__main__":
    scheduleMain(SHdayload["dayload"], SHdatas545, SHevfile, SHsensitivity_y)
