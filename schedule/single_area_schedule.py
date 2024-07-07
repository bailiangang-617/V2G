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
import cvxpy as cp
from cvxpy import Variable
from scipy.io import loadmat
import matplotlib.patches as patches
import pandas as pd

# 创建线程局部存储对象
thread_local = threading.local()
# 全局变量，用于记录线程编号
global_thread_id = 0
time = 96
time = 1344

# 充电负荷数据
df_charge_load = pd.read_csv(
    r"C:\V2G\Schedule\data\跨天调度\仿真数据处理\split_by_leid_count\1.csv_te_counts.csv"
)
day_ev = df_charge_load["count"]
#day_ev_array = day_ev.values[:time]
day_ev_array = day_ev.values
print(f"接入车辆数：{day_ev_array.shape}")
# evfile=loadmat(r'C:\Users\Administrator\Desktop\仿真数据处理\1_single_evfile.mat')
# evfile=loadmat(r'C:\Users\Administrator\Desktop\仿真数据处理\oneday_evfile_1.mat')
# evfile=pd.read_excel(r'C:\Users\Administrator\Desktop\仿真数据处理\oneday_evfile_1.xlsx',header=None)#必须要添加header等于None，否则后续转数组的时候，第一行的数据不会转过去
evfile = pd.read_csv(
    r"C:\V2G\Schedule\data\跨天调度\仿真数据处理\split_by_leid\1.csv"
)

lower_soc_file = pd.read_csv(r"C:\V2G\Schedule\data\demo1_lower副本.csv")
upper_soc_file = pd.read_csv(r"C:\V2G\Schedule\data\demo1_upper副本.csv")
# 电网数据
df_grid_load = pd.read_excel(
    r"C:\V2G\Schedule\data\电网数据处理\grid_load_preprogress.xlsx",
    engine="openpyxl",
)
grid_load = df_grid_load["电网负荷"]
#grid_load_array = grid_load.values[:time]
grid_load_array = grid_load.values
print(f"电网负荷长度：{grid_load_array.shape}")

def cvxSchedule(evtmp, Tcur, dayload,lower_soc_evtmp,upper_soc_evtmp):
    global global_thread_id

    # 获取或创建线程独立的存储对象
    if not hasattr(thread_local, "my_variable"):
        thread_local.my_variable = f"Thread_{global_thread_id}"
        global_thread_id += 1  # 每次创建一个新线程，全局线程编号加1
    # 现在可以使用 thread_local.my_variable，它对每个线程是独立的
    print(f"{thread_local.my_variable} - Thread ID: {threading.current_thread().name}")

    # 数据加载
    evtmp = evtmp[1:, :]  # 删除第一行，因为第一行为0，当时是为了拼接
    lower_soc_evtmp=lower_soc_evtmp[1:,:]
    upper_soc_evtmp=upper_soc_evtmp[1:,:]
    EVnum = evtmp.shape[0]  # 求行，有几行就有几辆调度的车
    print("可调度车辆", EVnum)
    a = 1
    b = 1
    Pchar = 30  # 最大充电功率
    Pdis = -30  # 最大放电功率
    Eini = evtmp[:, 1]  # 初始soc
    Efin = evtmp[:, 3]  # 目标电量
    Ecap = np.ones(EVnum) * 100  # 全部初始化为53.1
    Ezero = np.zeros(EVnum)
    Tleft = (evtmp[:, 2] - Tcur).astype(int)  # 剩余调度时间-->固定
    time = int(max(Tleft))
    print(f"time:{time}")
    #time = 96
    # 检查是否结束
    if time < 1:
        schedule = np.zeros((EVnum, 1))
        return schedule, 0
    Tcur = int(Tcur)
    print("调度开始时间：%d,可调度时间点：%d", Tcur, time)

    baseload = dayload[Tcur : Tcur + time]
    baseload = np.squeeze(baseload)  # 二维转一维
    # 凸优化
    z: Variable = Variable(time)
    x: Variable = Variable((EVnum, time))
    # batloss: Variable = Variable(time)
    objective = cp.Minimize(
        cp.sum(a * z + 0.5 * b * z**2 - a * baseload - 0.5 * b * baseload**2)
    )
    # 约束条件
    constraints = [
        z
        == baseload + cp.sum(x, axis=0)
    ]
    # 离网时间限制：可调度范围之外的充放电功率均为0
    for i in range(EVnum):
        if Tleft[i] < time:
            constraints.append(cp.constraints.Zero(x[i, Tleft[i]:time]))
    #充放电速率限制
    for i in range(EVnum):
        # if evtmp[4, i] == 1:#TODO:讨论V2G渗透率
        constraints += [Pdis <= x[i, :], x[i, :] <= Pchar]
    # else:
    #     constraints.append(0 <= x[i, :] <= Pchar)
    #每时刻的电量限制
    for i in range(time):
        constraints += [
            #Ezero <= Eini + cp.sum(x[:, : i + 1], axis=1),
            Eini + cp.sum(x[:, : i + 1], axis=1) <= Ecap,
        ]

    #离网电量限制
    for t in range(time):
        constraints += [
            Eini + cp.sum(x, axis=1)>=lower_soc_evtmp[:,Tcur+t+1:Tcur+t+2].flatten(),
            #Eini + cp.sum(x, axis=1) <= upper_soc_evtmp[:,Tcur+t+1:Tcur+t+2].flatten(),
            #Eini + cp.sum(x, axis=1) <= Ecap,
        ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status == cp.OPTIMAL:
            schedule = x[:, 0].value
            return schedule, problem.value
        else:
            # 处理问题状态不是最优的情况
            print(f"Problem status was {problem.status}, skipping this problem.")
            return None, None
    except cp.error.SolverError as e:
        # 求解器错误处理
        print(f"Solver failed with an error: {e}. Skipping this problem.")
    return None, None


def scheduleMain(dayload, thread_id, datas, evfile,lower_soc_file,upper_soc_file):
    # 参数设置
    time = 1344
    #time = 96
    tau = 0.25
    # evfile545 = evfile['evfile_demo_1']
    evfile545 = evfile.values[:time]  # 将数据转换成数组
    evfile545=evfile[evfile['te']<=time].values
    evnums=evfile545.shape[0]
    print(f"一天内的总行程数：{evnums}")
    lower_soc=lower_soc_file.values[:evnums]
    upper_soc=upper_soc_file.values[:evnums]
    EVnum = datas.sum()  # 这些地方要不要修改，修改后每次求解的范围可以减小
    scheLine = np.zeros((time, EVnum))  # 时间矩阵
    socLine = np.zeros((time, EVnum))  # 电量矩阵
    print(f"socline：{socLine[0,:].shape}")
    print(f"evfile545[:,1]:{evfile545[:,1].shape}")
    socLine[0, :] = evfile545[:, 1]  # 写入初始电量
    optvalLine = np.zeros(time)
    thread_local.scheLine = None
    thread_local.socLine = None
    for Tcur in range(time):
        ev_sum = datas[: Tcur + 1].sum()  # t时刻累计接入的电动汽车数
        print(f"在{Tcur}时刻，接入车辆数为{ev_sum}")
        if ev_sum > 0:
            evtmp = np.zeros((1, 19))  # 单辆EV出力矩阵
            count = 0
            lower_soc_evtmp=np.zeros((1, 1346))#这里这么大会不会影响效率
            upper_soc_evtmp=np.zeros((1, 1346))
            for i in range(ev_sum):  # 选择EV充放电策略，i表示车辆编号
                if (
                    evfile545[i, 2] >= Tcur and evfile545[i, 2] <= Tcur + 48 and evfile545[i, 2]<=time
                ):  # 离网时间大于当前时刻
                    evtmp = np.concatenate(
                        (evtmp, evfile545[i, :].reshape(1, -1)), axis=0#拼接的时候第一行的evtmp还是0，后续需要去掉
                    )  # 数据更新
                    lower_soc_evtmp= np.concatenate(
                        (lower_soc_evtmp, lower_soc[i, :].reshape(1, -1)), axis=0
                    )  # 数据更新
                    upper_soc_evtmp= np.concatenate(
                        (upper_soc_evtmp, upper_soc[i, :].reshape(1, -1)), axis=0
                    )  # 数据更新
                    count += 1
            if np.any(evtmp != 0):
                schedule, optval = cvxSchedule(evtmp, Tcur, dayload,lower_soc_evtmp,upper_soc_evtmp)
                optvalLine[Tcur] = optval  # 将优化值存放到optvalLine当前时刻位置
                #print(f"外面的schedule：",schedule)
                if schedule is not None:
                    point2 = 0
                    for i in range(ev_sum):
                        if (
                            evfile545[i, 2] >= Tcur and evfile545[i, 2] <= Tcur + 48 and evfile545[i, 2]<=time
                        ):  # 离网时间大于当前时刻，定位到调度的车辆
                            evfile545[i, 1] += schedule[point2]  # 更新电量
                            scheLine[Tcur, i] = schedule[point2]
                            point2 += 1
                            # print(i, evfile545[0, i])
                else:
                    print(
                        f"{thread_local.my_variable} - Failed to obtain optimal schedule."
                    )
                socLine[Tcur, :] = evfile545[:, 1]  # 记录电动汽车电量随时间的变化
    dayload = np.squeeze(dayload)
    z = dayload + np.sum(scheLine, axis=1)
    # 在这里保存计算结果到线程局部存储对象
    thread_local.scheLine = scheLine
    thread_local.socLine = socLine

    with pd.ExcelWriter(r"C:\V2G\Schedule\result\跨天调度\schedule_output_1.xlsx") as writer:
        # 将 SY_scheLine 保存到名为 'SY_scheLine' 的表格中
        df_scheLine = pd.DataFrame(scheLine)
        df_scheLine.to_excel(writer, sheet_name="SY_scheLine", index=False)

        # 将 SY_socLine 保存到名为 'SY_socLine' 的表格中
        df_socLine = pd.DataFrame(socLine)
        df_socLine.to_excel(writer, sheet_name="SY_socLine", index=False)

    # 优化曲线 
    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 使用中文字体
    plt.figure()
    plt.plot(dayload, "r")
    plt.plot(z, "b")
    plt.ylabel("负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "优化法接入EV后负荷"])
    plt.savefig(r"C:\V2G\Schedule\result\跨天调度\optimization1.png")
    plt.close()

    plt.figure()
    print(np.arange(time).shape)
    print(np.sum(scheLine, axis=1).shape)
    plt.bar(np.arange(time), np.sum(scheLine, axis=1), edgecolor="black")
    plt.ylabel("EV充放电负荷量 [KW]")
    plt.xlabel("时间区间")
    plt.legend(["电网基础负荷", "EV在电网中充放电量"])
    plt.savefig(r"C:\V2G\Schedule\result\跨天调度\EVload1.png")
    plt.close()
    return thread_local.scheLine, thread_local.socLine


# 多线程
def run_thread(dayload, thread_id, datas, evfile,lower_soc_file,upper_soc_file):
    thread_local.scheLine = None
    thread_local.socLine = None
    # scheduleMain(dayload, thread_id, datas, evfile, sensitivity_y)
    # return thread_local.scheLine, thread_local.socLine
    thread = threading.Thread(
        target=scheduleMain, args=(dayload, thread_id, datas, evfile,lower_soc_file,upper_soc_file)
    )
    thread.start()
    thread.join()  # 等待线程执行完毕
    #print(f"run_thread{thread_local.scheLine}")
    # scheLine, socLine=scheduleMain(dayload, thread_id, datas, evfile, sensitivity_y)
    return thread_local.scheLine, thread_local.socLine


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        # 提交任务并获取 Future 对象
        SY_future = executor.submit(
            run_thread, grid_load_array, 1, day_ev_array, evfile,lower_soc_file,upper_soc_file
        )
        # SH_future = executor.submit(run_thread, SHdayload['dayload'], 4, SHdatas545, SHevfile)
    # 获取各个区域线程的结果
    SY_scheLine, SY_socLine = SY_future.result()
    #print(f"amin{SY_scheLine}")
    # 创建一个 ExcelWriter 对象
    with pd.ExcelWriter("schedule_output_333.xlsx") as writer:
        # 将 SY_scheLine 保存到名为 'SY_scheLine' 的表格中
        df_scheLine = pd.DataFrame(SY_scheLine)
        df_scheLine.to_excel(writer, sheet_name="SY_scheLine", index=False)

        # 将 SY_socLine 保存到名为 'SY_socLine' 的表格中
        df_socLine = pd.DataFrame(SY_socLine)
        df_socLine.to_excel(writer, sheet_name="SY_socLine", index=False)
        # SH_scheLine, SH_socLine = SH_future.result()

    print("Processing results for SY area")
    # WUXUandRESULT_copy.data_compare(SY_scheLine, grid_load_array , 1)
