import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义函数来读取CSV文件并处理数据
def process_data(input_file, output_file1, output_file2, x_upeer_columns, x_lower_columns,y_upeer_columns,y_lower_columns):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    df1=pd.DataFrame(np.zeros((96, 96)))#为了使索引从0开始
    df2=pd.DataFrame(np.zeros((96, 96)))
    # 初始化两个空的DataFrame来保存插值结果

    # 遍历每一行数据
    for index, row in df.iterrows():
        #上限插值
        x_coords_upper = row[x_upeer_columns].values
        y_coords_upper = row[y_upeer_columns].values
        f = interp1d(x_coords_upper, y_coords_upper, kind='linear')
        x_int_upper = np.arange(int(np.ceil(x_coords_upper[0])), int(np.floor(x_coords_upper[-1])) + 1)
        y_int_upper = f(x_int_upper)
        for i in range(len(x_int_upper)):
            df1.loc[index,x_int_upper[i]]=y_int_upper[i]

        #下限插值
        x_coords_lower = row[x_lower_columns].values
        y_coords_lower  = row[y_lower_columns].values
        f = interp1d(x_coords_lower , y_coords_lower , kind='linear')
        x_int_lower  = np.arange(int(np.ceil(x_coords_lower [0])), int(np.floor(x_coords_lower[-1])) + 1)
        y_int_lower  = f(x_int_lower )
        for i in range(len(x_int_lower )):
            df2.loc[index,x_int_lower [i]]=y_int_lower [i]
        # 重新设置DataFrame的列名称，使其从0开始
        print(f"到第{index}行了，别着急")

    # 将零值替换为NaN
    df1.replace(np.nan,0, inplace=True)
    df2.replace(np.nan,0, inplace=True)
    df1.to_csv(output_file1,index=False)
    df2.to_csv(output_file2,index=False)


df = pd.read_csv(r"C:\V2G\Schedule\data\仿真数据处理\split_by_leid\1.csv")
input_file=r"C:\V2G\Schedule\data\不分天跨天调度\1副本.csv"
output_file1 = 'C:\V2G\Schedule\data\Zone1_lower_SOC_limit.csv'
output_file2 = 'C:\V2G\Schedule\data\Zone1_upper_SOC_limit.csv'
x_upeer_columns=['timea','timeb','timec']
x_lower_columns=['timea','timed','timee','timef']
y_upeer_columns=['soca','socb','socc']
y_lower_columns=['soca','socd','soce','socf']


#process_data(input_file, output_file1, output_file2, x_upeer_columns, x_lower_columns,y_upeer_columns,y_lower_columns)
# 读取保存的插值结果文件
df1 = pd.read_csv('C:\V2G\Schedule\data\Zone1_upper_SOC_limit.csv')

# 创建网格
X, Y = np.meshgrid(range(df1.shape[1]), range(df1.shape[0]))

# 创建绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维图形
ax.plot_surface(X, Y, df1.values, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Value')

# 显示图形
plt.show()
plt.savefig('C:\V2G\Schedule\result\Zone1_lower_SOC_limit.png')