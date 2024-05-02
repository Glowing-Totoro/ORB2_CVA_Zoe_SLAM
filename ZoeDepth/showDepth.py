import cv2


# 数据文件夹地址
DataPath = "/home/yu/dataset/test_show_depth_mvs/"
# 取图片名称
NumPhoto = 1
# 设置窗口
cv2.namedWindow("depth",0)
cv2.resizeWindow("depth",400,400)
    
while True:
    # 补0转换格式
    fixNumPhoto = '%06d' % NumPhoto 
    # 图片地址
    PhtotPath = DataPath + fixNumPhoto + '.png'
    print(PhtotPath)
    # 读取图片
    img = cv2.imread(PhtotPath)
    # 显示
    cv2.imshow('depth',img)
    if cv2.waitKey(5) == 27:
        break
    NumPhoto += 1
    if NumPhoto > 800:
        break
    # time.sleep(023.5)

cv2.destroyAllWindows()
