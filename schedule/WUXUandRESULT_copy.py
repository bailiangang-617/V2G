# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def data_compare(scheLine, dayload, thread_id):
    evfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
    evfile = evfile["evfile"]
    dayload = np.squeeze(dayload)  # 二维转一维
    price = loadmat(r"C:\V2G\Schedule\data\开源调度数据\PRICE.mat")
    timeprice = price["dongtaiprice"]
    sensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")

    # 汽车充电时段表示
    def carT(carnum, timeon, timeT):
        carT1 = np.zeros((96, carnum))

        for i in range(carnum):
            a = int(np.floor(timeon[i] / 0.25)) + 1
            b = int(np.floor((timeon[i] + timeT[i]) / 0.25)) + 1

            if a <= 96 and b <= 96:
                carT1[a - 1 : b, i] = 1
            elif a <= 96 and b > 96:
                carT1[a - 1 : 96, i] = 1
                carT1[0 : b - 96, i] = 1
            elif a > 96 and b > 96:
                carT1[a - 97 : b - 96, i] = 1

        return carT1

    # 电动汽车无序充电计算
    SOCend = (evfile[0, :] + 0.25 * np.sum(scheLine, axis=1)) / 53.1
    START = evfile[2, :] / 4
    SOCin = evfile[0, :] / 53.1
    idx1 = np.where(SOCin >= SOCend)[0]
    idx2 = np.where(SOCin < SOCend)[0]
    START1 = START[idx1]
    START2 = START[idx2]
    SOCend1 = SOCend[idx1]
    SOCend2 = SOCend[idx2]
    SOCin1 = SOCin[idx1]
    SOCin2 = SOCin[idx2]
    SOCend11 = SOCend[idx1]
    SOCend21 = SOCend[idx2]
    SOCin11 = SOCin[idx1]
    SOCin21 = SOCin[idx2]
    ET = 53.1

    # 全力无序充电
    timeT2 = np.zeros(len(SOCin2))
    for i in range(len(SOCin2)):
        while SOCin21[i] < SOCend21[i]:
            SOCin21[i] = SOCin21[i] + 0.25 * 7 / ET
            timeT2[i] = timeT2[i] + 0.25
        timeT2[i] = timeT2[i] - 0.25

    cartime2 = carT(len(SOCin2), START2, timeT2)
    A1 = 7 * cartime2

    # 全力无序放电
    timeT1 = np.zeros(len(SOCin1))
    for i in range(len(SOCin1)):
        while SOCend11[i] < SOCin11[i]:
            SOCend11[i] = SOCend11[i] + 0.25 * 7 / ET
            timeT1[i] = timeT1[i] + 0.25

    cartime1 = carT(len(SOCin1), START1, timeT1)
    A2 = -7 * cartime1

    # 将无序充放电功率整合起来
    P3 = np.sum(A1, axis=1) + np.sum(A2, axis=1) + dayload
    PP3 = np.sum(A1, axis=1) + np.sum(A2, axis=1)

    # 有序及无序功率曲线对比
    P1 = dayload
    P2 = np.sum(scheLine, axis=0) + dayload
    PP2 = sum(scheLine, 1)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure()
    plt.plot(P1, label="基础负荷")
    plt.plot(P3, label="无序充电")
    plt.ylabel("负荷/kW")
    plt.xlabel("时刻/15min")
    plt.legend()
    plt.title("无序充电")
    plt.savefig(r"C:\V2G\Schedule\result\开源数据调度\无序充电电网负荷.png")

    plt.figure()
    plt.plot(P1, label="基础负荷")
    plt.plot(P2, label="优化充放电")
    plt.ylabel("负荷/kW")
    plt.xlabel("时刻/15min")
    plt.legend()
    plt.title("优化调度")
    plt.savefig(r"C:\V2G\Schedule\result\开源数据调度\优化调度电网负荷.png")


    plt.show()

    # 计算峰谷差
    FGC1 = np.max(P1) - np.min(P1)
    FGC2 = np.max(P2) - np.min(P2)
    FGC3 = np.max(P3) - np.min(P3)

    print(f"thread_{thread_id}基础负荷峰谷差:", FGC1)
    print(f"thread_{thread_id}基础负荷峰值:", max(P1))
    print(f"thread_{thread_id}有序充放电峰谷差:", FGC2)
    print(f"thread_{thread_id}有序充电峰值:", np.max(P2))
    print(f"thread_{thread_id}无序充放电峰谷差:", FGC3)
    print(f"thread_{thread_id}无序充电峰值:", np.max(P3))

    # 验证计算结果正确
    print(f"thread_{thread_id}有序充放电总负荷:", np.sum(P2))
    print(f"thread_{thread_id}无序充放电总负荷:", np.sum(P3))

    # 电动汽车收益
    COST1 = np.sum(0.25 * PP2 * timeprice)
    COST2 = np.sum(0.25 * PP3 * timeprice)

    print(f"thread_{thread_id}场景一电动汽车充放电成本:", COST1)
    print(f"thread_{thread_id}场景二电动汽车充放电成本:", COST2)


if __name__ == "__main__":
    # 加载数据
    evfile = loadmat(r"C:\V2G\Schedule\data\开源调度数据\evfile545.mat")
    evfile = evfile["evfile"]
    jieguodata545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\jieguodata545.mat")
    scheLine = loadmat(r"C:\V2G\Schedule\data\开源调度数据\jieguodata545.mat")[
        "scheLine"
    ]
    datas545 = loadmat(r"C:\V2G\Schedule\data\开源调度数据\datas545.mat")
    dayload = datas545["dayload"]
    dayload = np.squeeze(dayload)  # 二维转一维
    price = loadmat(r"C:\V2G\Schedule\data\开源调度数据\PRICE.mat")
    timeprice = price["dongtaiprice"]
    sensitivity_y = loadmat(r"C:\V2G\Schedule\data\开源调度数据\sensitivity_y.mat")
    data_compare(scheLine, dayload, 1)
