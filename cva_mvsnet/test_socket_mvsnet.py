import argparse
import numpy as np
import torch
import random
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Tandem
from models.datasets import MVSDataset
from models.module import eval_errors
from models.utils.helpers import tensor2numpy, to_device
from models.utils import epoch_end_mean
import cv2
import pickle
import threading
import socket

# 接收到的图片
receive_photo = []
# 处理过的图片
process_photo = []
# 已发送的图片
send_photo = []
# 退出标志
Pro_Finished = False

# 主线程：循环扫描receive_photo和process_photo,只要有未处理图片即进行处理，记得延时0.1s，防止一直阻塞（一般不会）
# 副线程：通信，循环等待接受num，只要有就传递进入receive和process

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=str, help="Path to pytorch lightning ckpt.")
# 数据集的根目录路径
# parser.add_argument("--data_dir", help="Path to replica data.", type=str, default='data')
# 表示要保存用于可视化的图像的数量，默认值为 10
parser.add_argument("--num_save_images", help="Number of images to be saved for viz.", type=int, default=4)
# 表示随机数生成的种子，默认值为 1
parser.add_argument("--seed", help="Seed.", type=int, default=1)
# 选择使用的 PyTorch 设备，可选值为 'cpu' 或 'cuda'，默认为 'cuda'
parser.add_argument("--device", help="Torch device.", type=str, choices=('cpu', 'cuda'), default='cuda')
# 批处理的大小，默认值为 4，根据性能调整
parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
# 表示用于数据加载的工作进程数，默认值为0
parser.add_argument("--num_workers", help="Number of workers.", type=int, default=0)
# 表示元组的扩展名，默认值为 "dso_gs"
parser.add_argument("--tuples_ext", help="Tuples Extension.", type=str, default="dso_gs")
# 表示姿态文件的扩展名，默认值为 "dso"
parser.add_argument("--pose_ext", help="Pose Extension.", type=str, default="dso", choices=("dso", "gt"))
# 表示图像的高度，默认值为 480,这个根据实际数据集进行修改
parser.add_argument("--height", help="Image height.", type=int, default=480)
# 表示图像的宽度，默认值为 640,这个根据实际数据集进行修改
parser.add_argument("--width", help="Image width.", type=int, default=640)
# 一个可选参数 --depth_min，表示深度的最小值，默认值为 0.01,可调参
parser.add_argument("--depth_min", help="Depth minimum.", type=float, default=0.01)
# 一个可选参数 --depth_max，表示深度的最大值，默认值为 10.0,可调参
parser.add_argument("--depth_max", help="Depth maximum.", type=float, default=10.0)
# 表示数据集的分割文件，默认值为 "val"
parser.add_argument("--split", help="Split file", type=str, default="val")

def tcpServer():
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
    # 连接上,开始进行操作,接收到数字x则进行对应操作
    while True:
        # recv为阻塞操作,会一直卡在这一步
        data = my_server.recv(1024)  # 接收buffer的大小
        # 如果接受到空数据则退出
        if not data:
            break
        else:
            data = int(data)
            receive_photo.append(data)
            # time.sleep(1)        
        print("from connected user ", str(data))
        # data = str(data).upper()
        # print("sending data ", data)
        # my_server.send(data.encode())
    my_server.close()
    print(receive_photo)


def main(args: argparse.Namespace):

    thread1 = threading.Thread(target=tcpServer)
    thread1.start()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 这里设置载入数据所使用的线程数量，默认只是用主线程
    num_workers_for_data = 0

    device = torch.device(args.device)
    model = Tandem.load_from_checkpoint(args.ckpt)
    model = model.to(device)
    model.eval()
    outputs_to_dict = model.cva_mvsnet.outputs_to_dict
    batch_idx_keyframe = 10
    args.data_dir = "/home/yu/dataset_mvs/g/g"+str(batch_idx_keyframe)
    
    dataset = MVSDataset(
        root_dir=args.data_dir,
        split=args.split,
        pose_ext=args.pose_ext,
        tuples_ext=args.tuples_ext,
        ignore_pose_scale=args.pose_ext == "gt",
        height=args.height,
        width=args.width,
        tuples_default_flag=True,
        tuples_default_frame_num=2,
        tuples_default_frame_dist=1,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        dtype="float32",
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=num_workers_for_data)

    start = time.time()
    num_processed = 0
    cmpt_time = 0
    try:
        with torch.no_grad():
            while(batch_idx_keyframe<13):
                print("time of new loader:"+str(time.time()))
                data_dir = "/home/yu/dataset_mvs/g/g"+str(batch_idx_keyframe)
                # data_dir = "/mnt/e/Dataset/tandem_replica_1.1.beta/g/g"+str(batch_idx_keyframe)                
                dataset = MVSDataset(root_dir=data_dir,split=args.split,pose_ext=args.pose_ext,
                                     tuples_ext=args.tuples_ext,ignore_pose_scale=args.pose_ext == "gt",height=args.height,
                                     width=args.width,tuples_default_flag=True,tuples_default_frame_num=2,
                                     tuples_default_frame_dist=1,depth_min=args.depth_min,depth_max=args.depth_max,
                                     dtype="float32",transform=None,)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=num_workers_for_data)
                print(time.time())
                batch_idx = 0
                batch = next(iter(loader))
                print(time.time())
                cmpt_time = time.time()
                batch = to_device(batch, device=device)
                print(time.time())
                outputs = outputs_to_dict(model(batch))
                print(time.time())
                num_processed += args.batch_size
                print("the " + str(batch_idx) + " batch time:"+str(time.time()-cmpt_time))
                # 得到输出的预估深度图
                est = tensor2numpy(outputs['stage3']['depth'][0]).astype(np.float64) / args.depth_max
                if not np.all((est >= 0) & (est <= 1)):
                    print(f"Image out of bounds: min/max/median = {np.amin(est)}/{np.amax(est)}/{np.median(est)}")
                    est = np.clip(est, 0, 1)
                est = (est * float(np.iinfo(np.uint16).max)).astype(np.uint16)
                cv2.imwrite(data_dir+'/' +str(batch_idx_keyframe)+ '.png', est)
                batch_idx_keyframe += 1
    except KeyboardInterrupt:
        pass
    
    # 至此已经完成了所有的深度预估,后面就是对数据进行处理和存储
    # 计算时间指标
    elapsed = time.time() - start
    fps = num_processed / elapsed
    ms_per_frame = 1000.0 / fps
    print("num_processed:"+str(num_processed))
    print("ms_per_frame:"+str(ms_per_frame))


if __name__ == "__main__":
    main(parser.parse_args())
