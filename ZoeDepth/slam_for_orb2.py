import torch 
from PIL import Image
import tqdm 
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
import shutil 
import os
import glob 
import threading
import socket
import time
import cv2

receive_photo = []
process_photo = []
send_photo = []
Trans_Finished = False
Process_Finished = False
# 锁标志
lock = threading.Lock()

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
            Process_Finished == False
            lock.acquire()
            receive_photo.append(data)
            process_photo.append(-1)
            send_photo.append(-1)
            lock.release()
            # time.sleep(1)        
        if data == 200:
            Trans_Finished = True
        print("data from connected user ", str(data))
        # my_server.send(data.encode())
    my_server1.close()

def main():

    global Trans_Finished
    global Process_Finished
    # 主线程任务
    while(True):
        process = -1
        if Trans_Finished == True and Process_Finished == True:
            break
        # 取数据
        lock.acquire()
        Process_Finished = True
        if receive_photo:
            for i,item in enumerate(receive_photo):
                if process_photo[i]>=0:
                    continue
                else:
                    process = item
                    Process_Finished = False
                    break
        lock.release()                 
        if process >= 0:
            print("process:"+str(process))
            # 计算图片地址
            data_dir = "/home/yu/dataset/om1/"+str(process)+"/g/images/000001.png"
            output_dir = "/home/yu/dataset/om1/"+str(process)+"/g/depths/depth.png"
            # 处理并预测深度和存放
            try:
                image = Image.open(data_dir).convert("RGB")
            except:
                time.sleep(0.5)
                image = Image.open(data_dir).convert("RGB")
            depth = zoe.infer_pil(image, output_type="tensor", pad_input=False).cpu().detach().numpy()
            colored = colorize(depth)
            colored_img=Image.fromarray(colored)
            copy_for_colored = colored_img.copy()
            px = copy_for_colored.load()
            w, h = copy_for_colored.size
            for i in range(w):
                for j in range(h):
                    if type(px[i, j]) == int:
                        px[i, j] = 255-px[i, j]
                    elif len(px[i, j]) == 3:
                        px[i, j] = tuple([255-i for i in px[i, j]])
                    elif len(px[i, j]) == 4:
                        px[i, j] = tuple([255-i for i in px[i, j][:3]]+[px[i, j][-1]])
                    else:
                        pass
            copy_for_colored.save(output_dir)
            img = cv2.imread(output_dir)
            cv2.imshow('depth',img)
            if cv2.waitKey(1) == 27:
                break
            lock.acquire()
            process_photo[process] = 1
            try:
                my_server2.send(str(process).encode())
                send_photo[process] = 1
            except BrokenPipeError:
                print("Caught a BrokenPipeError,process Exited.")
                break
            print("process_photo:"+str(process))
            lock.release()
        else:
            time.sleep(0.1)

    print("receive_photo:")
    print(receive_photo)
    print("process_photo:")
    print(process_photo)
    print("send_photo")
    print(send_photo)
    
   
# 脚本入口,解析输入参数并执行main函数(本文件唯一函数)
if __name__ == "__main__":
    print("loading the model...")
    # 先加载好模型再通信
    conf = get_config("zoedepth", "infer")
    model_zoe_n = build_model(conf)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    print('zoe.device=', zoe.device)
    print("load finished,waitting to connect..")
    cv2.namedWindow("depth",0)
    cv2.resizeWindow("depth",400,400)
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
    main()