# 测试SLAM部分脚本
# TUM1配置 单目模式 固定词袋 最简数据集(来自rgbd_dataset_freiburg1_desk的纯图片部分)
# ./Examples/Monocular/mono_tum \
#     /home/yu/orbslam2_learn/ORB_SLAM2/Vocabulary/ORBvoc.txt\
#     Examples/Monocular/TUM1.yaml\
#     /home/yu/orbslam2_learn/Dataset/rgbd_dataset_freiburg1_desk_copy_test

# # 修改后
# ./Examples/Monocular/mono_tum \
#     Vocabulary/ORBvoc.txt\
#     Examples/Monocular/TUM1.yaml\
#     /home/yu/orbslam2_learn/Dataset/rgbd_dataset_freiburg1_xyz

# 修改后 ubutnu
# ./Examples/Monocular/mono_tum\
#     Vocabulary/ORBvoc.txt\
#     Examples/Monocular/TUM3.yaml\
#     /home/yu/dataset/orbslam_dataset/rgbd_dataset_freiburg3_long_office_household/rgbd_dataset_freiburg3_long_office_household

# # 修改后 ubutnu
# ./Examples/Monocular/mono_tum\
#     Vocabulary/ORBvoc.txt\
#     Examples/Monocular/TUM1.yaml\
#     /home/yu/dataset/orbslam_dataset/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk

# 制作数据集replica测试
# fr1
# ./Examples/Monocular/mono_tum\
#     Vocabulary/ORBvoc.txt\
#     Examples/Monocular/replica_1.yaml\
#     /home/yu/dataset/orbslam_dataset/frl_apartment_0

# a0
./Examples/Monocular/mono_tum\
    Vocabulary/ORBvoc.txt\
    Examples/Monocular/replica_1.yaml\
    /home/yu/dataset/orbslam_dataset/apartment_0
