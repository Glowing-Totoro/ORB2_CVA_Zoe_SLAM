import argparse
import numpy as np
import torch
import random
import time
import pickle
import threading
import socket

# 接收到的图片
receive_photo = []
# 处理过的图片
process_photo = []
# 已发送的图片
send_photo = []
# 传输完成标志
Trans_Finished = False
# 处理完成标志
Process_Finished = False
# 锁标志
lock = threading.Lock()

# 主线程：循环扫描receive_photo和process_photo,只要有未处理图片即进行处理，记得延时0.1s，防止一直阻塞（一般不会）
# 副线程：通信，循环等待接受num，只要有就传递进入receive和process

def tcpServer():
    global receive_photo
    global process_photo
    global Trans_Finished
    # 连接上,开始进行操作,接收到数字x则进行对应操作
    while True:
        # recv为阻塞操作,会一直卡在这一步
        data = my_server.recv(1024)  # 接收buffer的大小
        # 如果接受到空数据则退出
        if not data:
            break
        else:
            data = int(data)
            lock.acquire()
            receive_photo.append(data)
            process_photo.append(0)
            lock.release()
            # time.sleep(1)        
        if data == 20:
            Trans_Finished = True
        print("from connected user ", str(data))
        # my_server.send(data.encode())
    my_server.close()


# 程序开始
# 设置监听ip和端口
host = "127.0.0.1"
port = 5000
# 创建新的socket对象，并绑定到指定主机地址和端口
s = socket.socket()
s.bind((host, port))
# 让服务器开始监听传入的链接请求(参数1表示最多只能同时连接一个)
s.listen(1)  
# 接受客户端链接请求,accept会阻塞程序进行,直到连接上
my_server, address = s.accept()
print("connection from ", str(address))
thread1 = threading.Thread(target=tcpServer)
thread1.start()
while(True):
    process = 0
    if Trans_Finished == True and Process_Finished == True:
        break
    # 取数据
    lock.acquire()
    Process_Finished = True
    if receive_photo:
        for i,item in enumerate(receive_photo):
            if process_photo[i]:
                continue
            else:
                process = item
                Process_Finished = False
                break
    lock.release()            
    
    # 有待处理数据
    if process:
        # 模拟处理
        time.sleep(0.5)
        lock.acquire()
        process_photo[process-1] = 1
        my_server.send(str(process).encode())
        print("process_photo:"+str(process))
        lock.release()
    else:
        time.sleep(0.1)
print("receive_photo:")
print(receive_photo)
print("process_photo:")
print(process_photo)


