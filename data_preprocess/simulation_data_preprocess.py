import csv
import math
import os
import pandas as pd
from datetime import datetime


def filter_csv(input_file, output_file, threshold, column_name):
    with open(input_file, "r", newline="") as csv_input:
        with open(output_file, "w", newline="") as csv_output:
            reader = csv.reader(csv_input)
            writer = csv.writer(csv_output)

            # 读取标题行
            header = next(reader)
            writer.writerow(header)

            # 确定tp列的索引
            column_index = header.index(column_name)

            for row in reader:
                if int(row[column_index]) >= threshold:
                    writer.writerow(row)


def convert_to_timestamp(datetime_str):
    datatime_format = "%Y/%m/%d %H:%M"
    datetime_obj = datetime.strptime(datetime_str, datatime_format)
    timestamp = datetime_obj.timestamp()
    return timestamp


def convert_time_column(
    input_file,
    output_file,
    schedule_begin_timestamp,
    schedule_end_timestamp,
    time_intervel,
):
    df=pd.read_csv(input_file)
    # 删除包含缺失值的行
    # df.fillna("", inplace=True)
    #df.dropna(subset=['timea'], inplace=True)
    print(df)
    convert_to_timestamp_colums=['te','timea','timeb','timec','timed','timee','timef']
    for column in convert_to_timestamp_colums:
        df[column]=((df[column].astype(str).apply(convert_to_timestamp)-schedule_begin_timestamp)/time_intervel).astype(int)
    df['tp']=(df['te']+df["tp"].astype(int)*60/time_intervel).astype(int)
    df = df[(df['te'] <= schedule_end_timestamp) & (df['tp'] <= schedule_end_timestamp)]
    print(df)
    # # 写入到输出文件
    df.to_csv(output_file, index=False)

def count_te_values(csv_file):
    # 创建一个字典来记录 'te' 列中的值的计数
    te_counts = {str(i): 0 for i in range(1, 1345)}
    # 读取CSV文件并进行计数
    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            te_value = row.get("te")
            if te_value and te_value.isdigit() and 1 <= int(te_value) <= 1344:
                te_counts[te_value] += 1

    return te_counts


def write_count_to_csv(counts, output_file):
    # 写入计数结果到CSV文件
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["te_value", "count"])
        for te_value, count in counts.items():
            writer.writerow([te_value, count])


def split_and_count_csv(input_file, output_directory, count_output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(count_output_directory):
        os.makedirs(count_output_directory)

    df = pd.read_csv(input_file)
    grouped = df.groupby("Leid")

    # 遍历每个分组
    for leid, group in grouped:
        # 根据 'te' 列进行排序
        sorted_group = group.sort_values(by="te")
        output_file = os.path.join(output_directory, f"{leid}.csv")
        sorted_group.to_csv(output_file, index=False)

    # 对拆分后的CSV文件进行计数
    for csv_file in os.listdir(output_directory):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(output_directory, csv_file)
            te_counts = count_te_values(csv_path)
            count_output_file = os.path.join(
                count_output_directory, f"{csv_file}_te_counts.csv"
            )
            write_count_to_csv(te_counts, count_output_file)


if __name__ == "__main__":
    # 切割天的
    # input_file = r'C:\Users\Administrator\Desktop\仿真数据处理\仿真基准形成链数据.csv'  # 输入CSV文件名
    # filter_output_file = r'C:\Users\Administrator\Desktop\仿真数据处理\filter_output.csv'  # 输出CSV文件名
    # covert_time_output_file = r'C:\Users\Administrator\Desktop\帮师兄写毕业论文\仿真数据处理\covert_time_output_file.csv'
    # output_directory = r'C:\Users\Administrator\Desktop\仿真数据处理\split_by_leid'
    # output_count_directory = r'C:\Users\Administrator\Desktop\仿真数据处理\split_by_leid_count'
    # 不切割天的
    input_file = r"C:\Users\Administrator\Desktop\跨天调度\tripchain六边形.csv"  # 输入CSV文件名
    filter_output_file = r"C:\Users\Administrator\Desktop\跨天调度\filter_output.csv"  # 输出CSV文件名
    covert_time_output_file = r"C:\Users\Administrator\Desktop\跨天调度\仿真数据处理\covert_time_output_file.csv"
    output_directory = (
        r"C:\Users\Administrator\Desktop\跨天调度\仿真数据处理\split_by_leid"
    )
    output_count_directory = r"C:\Users\Administrator\Desktop\跨天调度\仿真数据处理\split_by_leid_count"
    schedule_begin_time = "2024/2/18 00:00"
    schedule_end_time = "2024/3/3 00:00"
    park_start_time_column_name = "te"
    park_duration_time_name = "tp"
    filter_column_name = "tp"
    threshold = 30
    time_intervel_minute = 15
    time_intervel_second = time_intervel_minute * 60
    schedule_begin_timestamp = convert_to_timestamp(schedule_begin_time)
    schedule_end_timestamp = convert_to_timestamp(schedule_end_time)
    print(schedule_begin_timestamp, schedule_end_timestamp)

    # 将停车时间小于30分钟的数据进行删除不进行调度处理
    filter_csv(input_file, filter_output_file, threshold, filter_column_name)

    # 将14天的停车时间以15分钟为一个时段进行拆分
    convert_time_column(
        filter_output_file,
        covert_time_output_file,
        schedule_begin_timestamp,
        schedule_end_timestamp,
        time_intervel_second,
    )

    # 按照117个区域划分成117个csv文件，并按照时间升序排序
    split_and_count_csv(
        covert_time_output_file, output_directory, output_count_directory
    )
    print("finished")
