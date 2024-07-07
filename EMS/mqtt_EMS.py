# -*- coding:utf-8 -*-
import time
import datetime
import json
import hashlib
from paho.mqtt import client as mqtt_client
import random
import sys
import re
import threading

# MQTT代理信息
mqtt_broker_ip = "113.200.194.123"
mqtt_port = 10311
# MQTT用户名和密码
mqtt_username = "your_username"
mqtt_password = "your_password"

# topic定义
mqtt_topic_LoginRequest = "C2S/station/1111111/reqLoginEvt"  # 7.2.1
mqtt_topic_LoginResponse = "S2C/station/1111111/respLoginSrv"  # 7.2.2
mqtt_topic_HeartBeat = "C2S/station/1111111/reqHeartBeatEvt"  # 7.2.3
mqtt_topic_HeartBeatResponse = "S2C/station/1111111/respHeartBeatEvt"  # 7.2.4
mqtt_topic_PlanScene = "S2C/station/1111111/reqSendPlanSceneEvt"  # 7.3.1
mqtt_topic_PlanSceneResponse = "C2S/station/1111111/respSendPlanSceneEvt"  # 7.3.2
mqtt_topic_TempScene = "C2S/station/1111111/reqSendTempSceneEvt"  # 7.4.1
mqtt_topic_TempSceneResponse = "S2C/station/1111111/respSendTempSceneEvt"  # 7.4.2
mqtt_topic_Baseload = "S2C/station/1111111/reqBaseloadEvt"  # 7.5.1
mqtt_topic_BaseloadResponse = "C2S/station/1111111/respBaseloadEvt"  # 7.5.2
mqtt_topic_SendSchedule = "C2S/station/1111111/reqSendScheduleEvt"  # 7.6.1
mqtt_topic_SendScheduleResponse = "S2C/station/1111111/respSendScheduleEvt"  # 7.6.2

# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
# 报文首部
device_code = 1111111  # 请替换为实际的站点编码


# 第一个是连接mqtt的函数，不需要改动
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            # 定义订阅的topic
            client.subscribe(mqtt_topic_HeartBeat)  # 心跳包
            client.subscribe(mqtt_topic_PlanScene)  # 订阅了7.3.1
            client.subscribe(mqtt_topic_PlanSceneResponse)  #7.3.2
            client.subscribe(mqtt_topic_TempScene)  # 订阅了7.4.1发送
            client.subscribe(mqtt_topic_TempSceneResponse)  # 订阅7.4.1接收
            client.subscribe(mqtt_topic_Baseload)  # 订阅了7.5.1
            client.subscribe(mqtt_topic_BaseloadResponse)  # 订阅了7.5.2
            client.subscribe(mqtt_topic_SendSchedule)  # 订阅了7.6.1

        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username=mqtt_username, password=mqtt_password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(mqtt_broker_ip, mqtt_port, 60)
    return client


# 第二个是将消息进行加密的函数，不需要改动/哈希值
def generate_data_sign(json_data_body, timestamp):
    combined_string = f"{json_data_body}:{timestamp}"
    utf8_bytes = combined_string.encode("utf-8")
    hash_object = hashlib.md5(utf8_bytes)
    hash_result = hash_object.hexdigest()
    return hash_result


# 第三个是发送消息的函数，需要根据实际情况进行修改，这个目前来说发送的格式是固定的，对于不同的指令，只需要修改data_body部分即可，databody不同的topic有专属的格式
def send_message(client, topic, data_body):
    timestamp = int(time.time() * 1000)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_data_body = re.sub(r':\s*', ':', json.dumps(data_body))
    json_data_body = re.sub(r',\s*', ',', json_data_body)
    data_sign = generate_data_sign(json_data_body, timestamp)

    message = {
        "head": {
            "deviceCode": device_code,
            "timeStamp": timestamp
        },
        "dataSign": data_sign,
        "dataBody": data_body
    }

    client.publish(topic, json.dumps(message, indent=None))
    print(f"Message sent: {json.dumps(message, indent=None)}")


# 定义发吉大系统发送函数-包括主动发送和回复发送
# 吉大主动发送
def send_heartbeat_periodically(client):  # 7.2.1
    while True:
        # 7.2.3 心跳上报
        send_message(client, mqtt_topic_HeartBeat, {"addValue": 1,
                                                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        time.sleep(10)


def send_temp_scene(client, scenetype):  # 7.4.1
    # 这个scenetype是由什么确定的不知道
    datalist = [1] * 1440  # datalist应该是根据零充给出的scentype来确定的，应该有个单独的函数来生成。这里先用长列表来替代
    datalist = [1]
    send_message(client, mqtt_topic_TempScene, {"sceneType": scenetype, "dataList": datalist,
                                                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


# 吉大接收后被动回复
def send_plan_scene_response(client, scenetype):  # 7.3.2
    resCode = 1  # 上报结果默认是成功
    datalist = [1] * 1440  # datalist应该是根据零充给出的scentype来确定的，应该有个单独的函数来生成。这里先用长列表来替代
    datalist = [1]
    send_message(client, mqtt_topic_PlanSceneResponse, {"sceneType": scenetype, "dataList": datalist,
                                                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        "resCode": resCode})


def send_baseload_response(client, scenetype):  # 7.5.2
    resCode = 1  # 上报结果默认是成功
    datalist = [1] * 1440  # datalist应该是根据零充给出的scentype来确定的，应该有个单独的函数来生成。这里先用长列表来替代
    datalist = [1]
    send_message(client, mqtt_topic_BaseloadResponse, {"sceneType": scenetype, "dataList": datalist,
                                                       "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                       "resCode": resCode})


# 定义收到信息后的回调函数
# 回调函数在接收到订阅主题的消息后就会触发，然后匹配消息主题，根据不同的主题，去调用不同的处理消息的函数
def on_message(client, userdata, msg):
    print("msg:", msg.topic)
    # 吉大先发送后EMS后回复
    if msg.topic == mqtt_topic_HeartBeat:  # 7.2.2
        try:
            payload = json.loads(msg.payload.decode())
            # 处理收到的场景指令数据
            print(f"Received mqtt_topic_HeartBeat message: {json.dumps(payload, indent=None)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    elif msg.topic == mqtt_topic_TempSceneResponse:  # 7.4.2
        try:
            payload = json.loads(msg.payload.decode())
            # 处理收到的场景指令数据
            print(f"Received mqtt_topic_TempSceneResponse message: {json.dumps(payload, indent=None)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    # 吉大EMS先发送吉大后回复
    elif msg.topic == mqtt_topic_PlanScene:  # 7.3.2
        try:
            payload = json.loads(msg.payload.decode())  # 这一步的作用是将收到的消息转换称为python对象
            # 处理收到的场景指令数据，这里应该是解析出计划电网场景scenetype的编号，然后调用（7.3.2吉大系统回复电网场景指令）这个向EMS回复的函数
            process_send_plan_scene_event(client, payload)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    elif msg.topic == mqtt_topic_Baseload: # 7.5.2
        try:
            payload = json.loads(msg.payload.decode())
            # 处理收到的场景指令数据
            process_send_base_load_event(client, payload)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Ignored message from unexpected topic '{msg.topic}'")


# 定义接收到EMS信息后的处理函数
def process_send_plan_scene_event(client, payload):  # 7.3.1处理
    # 为了方便调试，这里先打印出收到的消息
    print(f"Received SendPlanSceneEvt message: {json.dumps(payload, indent=None)}")

    # 后续得添加逻辑，获取scenetype，然后调用（7.3.2吉大系统回复电网场景指令）这个向EMS回复的函数
    scenetype = payload["dataBody"]  # 这个是我猜的,得根据实际的收到的数据去修改，调试时候需要先注释掉这行在内的后两行
    send_plan_scene_response(client, scenetype)


def process_send_base_load_event(client, payload):  # 7.5.1处理
    # 为了方便调试，这里先打印出收到的消息
    print(f"Received SendBaseLoadEvt message: {json.dumps(payload, indent=None)}")
    # 后续得添加逻辑，获取scenetype，然后调用（7.5.2吉大系统回复基础负荷指令）这个向EMS回复的函数
    scenetype = payload["dataBody"]  # 这个是我猜的,得根据实际的收到的数据去修改，调试时候需要先注释掉这行在内的后两行
    send_baseload_response(client, scenetype)


# 下边定义一个运行函数，当程序开始执行的时候就是运行这个主函数
def run():
    # 连接到MQTT代理
    client = connect_mqtt()
    # 启动一个新的线程来运行send_heartbeat_periodically函数，对应了7.2.3的心跳上报
    heartbeat_thread = threading.Thread(target=send_heartbeat_periodically, args=(client,))  # 7.2.1心跳包发送
    temp_scene_thread = threading.Thread(target=send_temp_scene(client, 1), args=(client,))  # 7.4.1发送
    heartbeat_thread.start()
    temp_scene_thread.start()

    # 在这里可以启动其他线程来运行不同的功能，例如发送计划性电网场景、临时性电网场景、基础负荷、调度指令等
    # 这里的其他功能是指什么时候接收用户的输入，然后给EMS发送对应的指令（这部分还没有想好，也是代码目前联合调试的盲点，因为我们一直是空中楼阁，还没有完整的软件集成逻辑）

    # 保持脚本运行的主循环
    client.loop_forever()


# 定义主函数
if __name__ == '__main__':
    run()
