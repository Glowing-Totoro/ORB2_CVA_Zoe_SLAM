# unset LD_LIBRARY_PATH
# 导入必要的库与模块

# 用于命令行参数的模块
import argparse
# 经典的numpy库
import numpy as np
# pytorch
import torch
# 生成伪随机数
import random
# 时间操作处理
import time
# 显示进度条
from tqdm import tqdm
# 从PyTorch中导入用于加载数据的 DataLoader 类
from torch.utils.data import DataLoader
# 从自定义模块 models 中导入类和函数
from models import Tandem
from models.datasets import MVSDataset
from models.module import eval_errors
from models.utils.helpers import tensor2numpy, to_device
from models.utils import epoch_end_mean
# opencv
import cv2
# 用于序列化和反序列化
import pickle
# 通信与多线程
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

parser = argparse.ArgumentParser()

parser.add_argument("ckpt", type=str, help="Path to pytorch lightning ckpt.")
parser.add_argument("--num_save_images", help="Number of images to be saved for viz.", type=int, default=4)
parser.add_argument("--seed", help="Seed.", type=int, default=1)
parser.add_argument("--device", help="Torch device.", type=str, choices=('cpu', 'cuda'), default='cuda')
parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
parser.add_argument("--num_workers", help="Number of workers.", type=int, default=0)
parser.add_argument("--tuples_ext", help="Tuples Extension.", type=str, default="dso_gs")
parser.add_argument("--pose_ext", help="Pose Extension.", type=str, default="dso", choices=("dso", "gt"))
parser.add_argument("--height", help="Image height.", type=int, default=480)
parser.add_argument("--width", help="Image width.", type=int, default=640)
parser.add_argument("--split", help="Split file", type=str, default="val")

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
        if data == 100:
            Trans_Finished = True
        print("data from connected user ", str(data))
        # my_server.send(data.encode())
    my_server1.close()

def main(args: argparse.Namespace):

    global Trans_Finished
    global Process_Finished

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = Tandem.load_from_checkpoint(args.ckpt)
    # 这里加载模型是最为耗时的，仅加载一次即可
    model = model.to(device)
    model.eval()
    outputs_to_dict = model.cva_mvsnet.outputs_to_dict
    
    # 自调整参数部分
    
    # 最大与最小深度
    depth_max_set = 10
    depth_min_set = 0.01
    # 首次处理文件idx
    batch_idx_keyframe = 0
    # batchsize
    tuples_frame_num = 4
    tuples_dist_num = 1
    # 加载进程数量
    num_workers_for_data = 0
    
    # 主线程任务
    while(True):
        batch_idx_keyframe = 0
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
    
        # 有待处理数据
        if process>=0:
            # 深度估计
            # start
            start = time.time()
            num_processed = 0
            try:
                with torch.no_grad():
                    while(batch_idx_keyframe<1):
                        print("time of new loader:"+str(time.time()))
                        data_dir = "/home/yu/dataset/om1/"+str(process)                
                        dataset = MVSDataset(root_dir=data_dir,split=args.split,pose_ext=args.pose_ext,
                                     tuples_ext=args.tuples_ext,ignore_pose_scale=args.pose_ext == "gt",height=args.height,
                                     width=args.width,tuples_default_flag=True,tuples_default_frame_num=tuples_frame_num,
                                     tuples_default_frame_dist=tuples_dist_num,depth_min=depth_min_set,depth_max=depth_max_set,
                                     dtype="float32",transform=None,)
                        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers_for_data)
                        print(time.time())
                        batch_idx = 0
                        batch = next(iter(loader))
                        print(time.time())
                        batch = to_device(batch, device=device)
                        print(time.time())
                        outputs = outputs_to_dict(model(batch))
                        print(time.time())
                        num_processed += 1
                        # 预测深度图
                        image = tensor2numpy(outputs['stage3']['depth'][0]).astype(np.float64) / depth_max_set
                
                        if not np.all((image >= 0) & (image <= 1)):
                            print(f"Image out of bounds: min/max/median = {np.amin(image)}/{np.amax(image)}/{np.median(image)}")
                            image = np.clip(image, 0, 1)
                        # 像素值范围转化,从 [0, 1] 转换为 [0, 2^16-1] 的范围
                        image = (image * float(np.iinfo(np.uint16).max)).astype(np.uint16)
                        outputs_str = data_dir +'/g/depths/depth.png'
                        cv2.imwrite(outputs_str, image)
                        cv2.imshow('depth',image)
                        if cv2.waitKey(1) == 27:
                            break
                        batch_idx_keyframe += 1
            except KeyboardInterrupt:
                pass        
        
        
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
    # my_server2.close()
    cv2.destroyAllWindows()
    print("receive_photo:")
    print(receive_photo)
    print("process_photo:")
    print(process_photo)
    print("send_photo")
    print(send_photo)
    
   
# 脚本入口,解析输入参数并执行main函数(本文件唯一函数)
if __name__ == "__main__":
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
    main(parser.parse_args())