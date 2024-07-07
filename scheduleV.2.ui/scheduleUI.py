# -*- coding:utf-8 -*-
import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5 import uic
from PyQt5 import QtCore
from isapi.threaded_extension import WorkerThread
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib_inline.backend_inline import FigureCanvas
from scipy.io import loadmat
import schedule_main
import icon


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载UI文件
        uic.loadUi("scheduleV.2.ui", self)
        self.pushButton_9.clicked.connect(self.on_button_clicked)
        self.progress_bar = self.progressBar
        self.progress_bar.setValue(0)  # 初始化进度条；不初始化就会闪退
        # 获取当前脚本文件所在的目录
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # # 构建图标文件的绝对路径
        # icon_path = os.path.join(script_dir, '吉大校徽.jpg')

        # # 将路径替换为你的图标路径
        # icon = QIcon(icon_path)
        # self.setWindowIcon(icon)

    def on_button_clicked(self):
        # 处理按钮点击的逻辑
        selected_item_senses = self.comboBox_2.currentText()
        selected_item_EV = self.comboBox.currentText()
        self.worker_thread = WorkerThread()
        self.worker_thread.progress_updated.connect(
            self.update_progress_bar
        )  # 连接更新进度条的信号
        self.worker_thread.result_ready.connect(self.handle_thread_result)
        # 根据选择项调用对应的函数
        if selected_item_senses == "小区" and selected_item_EV == "测试例_545":
            print("来了来了")
            self.worker_thread.start()

    def handle_thread_result(self, result):
        dayload, znow, scheLine, sensitivity_y, datas545, expenses, FGC, Tcur = result
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        fig = Figure(figsize=(5.5, 4))
        ax = fig.add_subplot(111)
        ax.plot(dayload, "r")
        ax.plot(znow, color="#4CAF50")
        ax.set_ylabel("负荷量 [KW]", color="#FFFFFF")
        ax.set_xlabel("时间区间", color="#FFFFFF")
        ax.legend(["电网基础负荷", "优化法接入EV后负荷"])

        # 坐标美化
        # ax.plot([0, 96], [0, 0], color='white', linewidth=2)  # 横坐标为白色
        ax.set_facecolor("#001E2F")
        fig.set_facecolor("#001E2F")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("#ffffff")
        ax.spines["bottom"].set_color("#ffffff")
        ax.tick_params(axis="both", colors="#FFFFFF")
        ax.axhline(y=0, color="white", linewidth=2)
        ax.set_xlim([0, 100])  # 设置 x 轴范围为 0 到 6
        ax.set_ylim([1800, 3200])  # 设置 y 轴范围为 0 到 6

        # 将 Matplotlib 图形转换为 QPixmap
        pixmap = self.fig_to_pixmap(fig)
        self.label_15.setStyleSheet("")  # 清除样式表
        self.label_15.repaint()  # 刷新界面
        # 将 QPixmap 设置到 QLabel 中
        self.label_14.setPixmap(pixmap)
        self.label_14.setAlignment(Qt.AlignCenter)  # 布局中间

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        fig = plt.gcf()  # 获取当前的Figure对象
        fig.clf()  # 清除之前的图形内容
        ax = fig.gca()
        plt.bar(
            np.arange(96), np.sum(scheLine, axis=0), edgecolor="black", color="#4CAF50"
        )
        plt.ylabel("EV充放电负荷量 [KW]", color="#FFFFFF")
        plt.xlabel("时间区间", color="#FFFFFF")
        plt.legend(["EV充放电量"])

        ax.set_facecolor("#001E2F")  # 修改图片背景
        fig.set_facecolor("#001E2F")  # 修改窗口背景
        ax.spines["right"].set_visible(False)  # 删除右边的坐标线
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("#ffffff")  # 修改左坐标线的颜色
        ax.spines["bottom"].set_color("#ffffff")  # 修改坐标刻度的颜色
        ax.tick_params(axis="both", colors="#FFFFFF")
        ax.axhline(y=0, color="white", linewidth=1)  # 修改y=0的线段的颜色
        fig.set_size_inches(5.5, 4)  # 设置图片尺寸

        ax.set_xlim([0, 96])  # 设置 x 轴范围
        ax.set_ylim([-300, 300])  # 设置 y 轴范围

        pixmap = self.fig_to_pixmap(fig)
        self.label_15.setStyleSheet("")  # 清除样式表
        self.label_15.repaint()  # 刷新界面
        # 将 QPixmap 设置到 QLabel 中
        self.label_15.setPixmap(pixmap)
        self.label_15.setAlignment(Qt.AlignCenter)  # 布局中间

        self.label_32.setText(f"{expenses:.2f}")
        self.label_33.setText(f"{FGC:.2f}")
        self.label_34.setText(f"{sum(znow) * 0.25:.2f}")
        self.label.setText(f"调度时段：{Tcur + 1:}")

    def fig_to_pixmap(self, fig):
        # 将 Matplotlib 图形转换为 QPixmap
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()

        # 获取图形数据
        buf = canvas.buffer_rgba()

        # 使用 QImage 创建 QPixmap
        qimage = QImage(buf, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        return pixmap

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)


class WorkerThread(QThread):
    result_ready = pyqtSignal(tuple)
    progress_updated = pyqtSignal(int)

    def run(self):
        # 获取当前脚本文件所在的目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(script_dir)

        # 构建.mat文件的绝对路径
        datas545_path = os.path.join(script_dir, "data", "datas545.mat")
        print(datas545_path)
        evfile_path = os.path.join(script_dir, "data", "evfile545.mat")
        sensitivity_y_path = os.path.join(script_dir, "data", "sensitivity_y.mat")

        # 尝试加载.mat文件
        try:
            datas545 = loadmat(datas545_path)
            evfile = loadmat(evfile_path)
            sensitivity_y = loadmat(sensitivity_y_path)
        except FileNotFoundError:
            print("Error: One or more files not found.")
        evfile545 = evfile["evfile"]
        # 参数设置
        time = 96
        tau = 0.25
        expenses = 0  # 充电成本
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
            schedule, optval = schedule_main.cvxSchedule(evtmp, Tcur)

            optvalLine[Tcur] = optval  # 将优化值存放到optvalLine当前时刻位置
            if schedule is not None:
                print(f"Optimal Schedule: {schedule}")
                point2 = 0
                for i in range(point - 1):
                    if (
                        evfile545[3, i] >= Tcur
                    ):  # 离网时间大于当前时刻，定位到调度的车辆
                        evfile545[0, i] += schedule[point2]  # 更新电量
                        scheLine[i, Tcur] = schedule[point2]
                        point2 += 1
            else:
                print("Failed to obtain optimal schedule.")
            socLine[:, Tcur + 1] = evfile545[0, :]  # 记录电动汽车电量随时间的变化
            price = np.squeeze(datas545["timeprice"])
            expenses += np.sum(scheLine[:, Tcur]) * price[Tcur] * 0.25
            # 计算当前迭代的进度百分比
            progress_percentage = int((Tcur + 1) / time * 100)
            self.progress_updated.emit(progress_percentage)
            dayload = np.squeeze(datas545["dayload"])
            znow = dayload[: Tcur + 1] + np.sum(scheLine[:, : Tcur + 1], axis=0)
            FGC = np.ptp(znow)
            self.result_ready.emit(
                (dayload, znow, scheLine, sensitivity_y, datas545, expenses, FGC, Tcur)
            )
            # self.scheLine = scheLine


if __name__ == "__main__":
    app = QApplication([])
    window = MyMainWindow()
    window.show()
    app.exec_()
