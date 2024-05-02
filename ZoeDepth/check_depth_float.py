import cv2
import numpy as np

def showdiff(dir):
    print("dir:")
    depth_image = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
    max_depth1 = np.max(depth_image)
    mean_depth1 = np.mean(depth_image)
    print("max: ",max_depth1," in unchanged")
    print("mean: ",mean_depth1," in unchanged")    
    depth_image = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    max_depth2 = np.max(depth_image)
    mean_depth2 = np.mean(depth_image)
    print("max: ",max_depth2," in gray")   
    print("mean: ",mean_depth2," in gray")    


dir= '/home/yu/dataset/om1/0/g/depths/tum1.png'
showdiff(dir)
dir= '/home/yu/dataset/om1/0/g/depths/000000.png'
showdiff(dir)
dir= '/home/yu/dataset/om1/0/g/depths/depth.png'
showdiff(dir)






if __name__ == '__main__':
    invert_color('test.jpg')



# # 遍历深度图像的每个像素并输出深度值
# for row in range(depth_image.shape[0]):
#     for col in range(depth_image.shape[1]):
#         depth_value = depth_image[row, col]
#         print("深度值(row={}, col={}): {}".format(row, col, depth_value))
