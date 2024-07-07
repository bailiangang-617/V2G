# -*- coding:utf-8 -*-
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy import Variable
from scipy.io import loadmat
import matplotlib.patches as patches

# import WUXUandRESULT_copy
from concurrent.futures import ThreadPoolExecutor


# 创建线程局部存储对象
thread_local = threading.local()
# 全局变量，用于记录线程编号
global_thread_id = 0
# 商业区--10个
SYdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
SYevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
SYsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
SYdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\SYdayload.mat")

# 办公区--轻工业10个
BGdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
BGevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
BGsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
BGdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\BGdayload.mat")

# 写字楼--10个
XZdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
XZevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
XZsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
XZdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\XZdayload.mat")

# 生活区--原始数据
SHdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
SHevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
SHsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
SHdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\SHdayload.mat")

# 重工业--10个
ZGYdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
ZGYevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
ZGYsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
ZGYdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\ZGYdayload.mat")

# 科研机构--5个
KYdatas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
KYevfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
KYsensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
KYdayload = loadmat(r"C:\V2G\Schedule\data\六场景多线程调度\KYdayload.mat")


def cvxSchedule(evtmp, Tcur, dayload, sensitivity_y):
    global global_thread_id

    # 获取或创建线程独立的存储对象
    if not hasattr(thread_local, "my_variable"):
        thread_local.my_variable = f"Thread_{global_thread_id}"
        global_thread_id += 1  # 每次创建一个新线程，全局线程编号加1
    # 现在可以使用 thread_local.my_variable，它对每个线程是独立的
    print(f"{thread_local.my_variable} - Thread ID: {threading.current_thread().name}")

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


def scheduleMain(dayload, thread_id, datas, evfile, sensitivity_y):
    # 参数设置
    time = 96
    tau = 0.25
    evfile545 = evfile["evfile"]
    EVnum = datas["dayev"].sum()
    scheLine = np.zeros((EVnum, time))  # 时间矩阵
    socLine = np.zeros((EVnum, time + 1))  # 电量矩阵
    socLine[:, 0] = evfile545[0, :]  # 写入初始电量
    optvalLine = np.zeros(time)
    thread_local.scheLine = None
    thread_local.socLine = None
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
            print(f"{thread_local.my_variable} - Failed to obtain optimal schedule.")
        socLine[:, Tcur + 1] = evfile545[0, :]  # 记录电动汽车电量随时间的变化

    dayload = np.squeeze(dayload)
    z = dayload + np.sum(scheLine, axis=0)
    # 在这里保存计算结果到线程局部存储对象
    thread_local.scheLine = scheLine
    thread_local.socLine = socLine

    # 结果保存
    df_z = pd.DataFrame(z)
    df_z.to_csv(
        rf"C:\V2G\Schedule\result\六区域多线程调度\thread_{thread_id}_output_z.csv",
        index=False,
        header=False,
    )

    df_socLine = pd.DataFrame(socLine)
    df_socLine.to_csv(
        rf"C:\V2G\Schedule\result\六区域多线程调度\thread_{thread_id}_output_socLine.csv",
        index=False,
        header=False,
    )

    df_scheLine = pd.DataFrame(scheLine)
    df_scheLine.to_csv(
        rf"C:\V2G\Schedule\result\六区域多线程调度\thread_{thread_id}_output_scheLine.csv",
        index=False,
        header=False,
    )

    # 优化曲线
    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 使用中文字体
    plt.figure()
    plt.plot(dayload, "r")
    plt.plot(z, "b")
    plt.ylabel("负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "优化法接入EV后负荷"])
    plt.savefig(
        rf"C:\V2G\Schedule\result\六区域多线程调度\thread_{thread_id}_optimization.png"
    )

    plt.figure()
    plt.bar(np.arange(time), np.sum(scheLine, axis=0), edgecolor="black")
    plt.ylabel("EV充放电负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "EV在电网中充放电量"])
    plt.savefig(
        rf"C:\V2G\Schedule\result\六区域多线程调度\thread_{thread_id}_EVload.png"
    )
    # return thread_local.scheLine, thread_local.socLine


# 多线程
def run_thread(dayload, thread_id, datas, evfile, sensitivity_y):
    thread_local.scheLine = None
    thread_local.socLine = None
    # scheduleMain(dayload, thread_id, datas, evfile, sensitivity_y)
    # return thread_local.scheLine, thread_local.socLine
    thread = threading.Thread(
        target=scheduleMain, args=(dayload, thread_id, datas, evfile, sensitivity_y)
    )
    thread.start()
    thread.join()  # 等待线程执行完毕
    # scheLine, socLine=scheduleMain(dayload, thread_id, datas, evfile, sensitivity_y)
    return thread_local.scheLine, thread_local.socLine


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        # 提交任务并获取 Future 对象
        SY_future = executor.submit(
            run_thread, SYdayload["DAYLOAD"], 1, SYdatas545, SYevfile, SYsensitivity_y
        )
        BG_future = executor.submit(
            run_thread, BGdayload["BGdayload"], 2, BGdatas545, BGevfile, BGsensitivity_y
        )
        XZ_future = executor.submit(
            run_thread, XZdayload["XZdayload"], 3, XZdatas545, XZevfile, XZsensitivity_y
        )
        SH_future = executor.submit(
            run_thread, SHdayload["dayload"], 4, SHdatas545, SHevfile, SHsensitivity_y
        )
        ZGY_future = executor.submit(
            run_thread,
            ZGYdayload["ZGYdayload"],
            5,
            ZGYdatas545,
            ZGYevfile,
            ZGYsensitivity_y,
        )
        KY_future = executor.submit(
            run_thread, KYdayload["KYdayload"], 6, KYdatas545, KYevfile, KYsensitivity_y
        )

    # 获取各个区域线程的结果
    SY_scheLine, SY_socLine = SY_future.result()
    BG_scheLine, BG_socLine = BG_future.result()
    XZ_scheLine, XZ_socLine = XZ_future.result()
    SH_scheLine, SH_socLine = SH_future.result()
    ZGY_scheLine, ZGY_socLine = ZGY_future.result()
    KY_scheLine, KY_socLine = KY_future.result()

    # # 在这里进行结果的处理或其他操作
    # # 例如，可以在这里调用 WUXUandRESULT_copy.data_compare 函数，并在前面添加 print 语句输出线程运行信息
    # print("Processing results for SY area")
    # WUXUandRESULT_copy.data_compare(SY_scheLine, SYdayload['DAYLOAD'], 1)

    # print("Processing results for BG area")
    # WUXUandRESULT_copy.data_compare(BG_scheLine, BGdayload['dayload'], 2)
