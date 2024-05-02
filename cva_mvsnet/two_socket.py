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

def ReceiveServer():
    global receive_photo
    global process_photo
    global send_photo
    global Trans_Finished
    # 连接上,开始进行操作,接收到数字x则进行对应操作
    while True:
        # recv为阻塞操作,会一直卡在这一步
        data = my_server1.recv(1024)  # 接收buffer的大小
        # 如果接受到空数据则退出
        if not data:
            break
        else:
            data = int(data)
            lock.acquire()
            receive_photo.append(data)
            process_photo.append(0)
            send_photo.append(0)
            lock.release()
            # time.sleep(1)        
        if data == 20:
            Trans_Finished = True
        print("from connected user ", str(data))
        # my_server.send(data.encode())
    my_server1.close()


# 程序开始
    
# socket 1
# 设置监听ip和端口
host = "127.0.0.1"
port1 = 5001
# 创建新的socket对象，并绑定到指定主机地址和端口
s1 = socket.socket()
s1.bind((host, port1))
# 让服务器开始监听传入的链接请求(参数1表示最多只能同时连接一个)
s1.listen(1)  
# 接受客户端链接请求,accept会阻塞程序进行,直到连接上
my_server1, address1 = s1.accept()
print("socket1 connection from ", str(address1))
# socket 2
port2 = 5002
s2 = socket.socket()
s2.bind((host, port2))
s2.listen(1)  
my_server2, address2 = s2.accept()
print("socket2 connection from ", str(address2))

# 线程设置:thread1为接收，thread2为发送（主线程）
thread1 = threading.Thread(target=ReceiveServer)
thread1.start()


# 主线程任务
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
        try:
            my_server2.send(str(process).encode())
            send_photo[process-1] = 1
        except BrokenPipeError:
            print("Caught a BrokenPipeError,process Exited.")
            break
        print("process_photo:"+str(process))
        lock.release()
    else:
        time.sleep(0.1)
my_server2.close()
print("receive_photo:")
print(receive_photo)
print("process_photo:")
print(process_photo)
print("send_photo")
print(send_photo)
