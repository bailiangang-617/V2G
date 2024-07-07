import pandas as pd
import matplotlib.pyplot as plt

lower_soc_file = pd.read_csv(r"C:\V2G\Schedule\data\demo1_lower副本.csv")
upper_soc_file = pd.read_csv(r"C:\V2G\Schedule\data\demo1_upper副本.csv")

#读取csv文件
def read_row_from_csv(file_path,row_index):
    data=pd.read_csv(file_path)
    row = data.iloc[row_index]
    # 筛选出非零值
    non_zero_row = row[row != 0]
    return non_zero_row

# 绘制函数
def plot_data(x, y1, y2, labels):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=labels[0], marker='o',color='b')  # 绘制第一个文件的数据
    plt.plot(x, y2, label=labels[0], marker='o',color='b')  # 绘制第二个文件的数据
    plt.plot([x[-1], x[-1]], [y1[-1], y2[-1]], label=labels[0], marker='o',color='b')  
    plt.ylabel('SOC')
    plt.xlabel('T')
    plt.title('SOC scheduling scope')
    #plt.legend()
    plt.grid(True)
    plt.show()

# 文件路径
file1 = "C:\V2G\Schedule\data\demo1_lower副本.csv"
file2 = "C:\V2G\Schedule\data\demo1_upper副本.csv"

# 选择要读取的行索引
row_index = 25  # 例如读取第一行数据

# 读取数据
row_data1 = read_row_from_csv(file1, row_index)
row_data2 = read_row_from_csv(file2, row_index)

# 准备绘图数据
x_labels = row_data1.index.tolist()  # 假设每列的标题是相同的
y_data1 = row_data1.values
y_data2 = row_data2.values

# 绘制数据
plot_data(x_labels, y_data1, y_data2, ['File 1', 'File 2'])
