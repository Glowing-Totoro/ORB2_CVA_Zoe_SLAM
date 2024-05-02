# 过程记录

### 3/18计划工作

1. 明确MVSnet需求的数据集格式，制作合适的数据集测试
2. 测试不同步骤的执行效率(时间)，找到载入数据集数据的代码
3. 测试py文件内改变数据参数，对多个自行制作的数据集进行测试
4. 自行设计测试py文件，除去冗余步骤
5. 设置评估代码，明确导出数据格式

### 3/18实际工作

1. 检查pytorch小bug并除去，暂时未发现异常，运行效率未变化（在/home/yu/anaconda3/envs/cvamvs/lib/python3.7/site-packages/torch/functional.py:445添加`,indexing = 'ij'`,去除`torch.meshgrid`错误）
2. 明确了每个步骤的执行时间，实际上加载模型是最耗费时间的，不过这只执行一次，在单目初始化中执行即可，此外每次加载数据和模型计算最为耗时，为了提高执行速度，保证实时性，减少了每张图片估计次数，只执行一次而去掉循环，并且减小加载数据的线程数量，将`num_workers=0`设置为0，只利用主线程的速度反而是最快的（这是因为每个数据集非常小），最后帧率基本达到2FPS以上，基本达到了要求
3. 剩余任务：数据集输入和输出的格式细致调查(1 5)

### 3/19计划工作

1. 明确MVSnet输入数据格式与输出数据格式，orbslam2输入数据格式与输出数据格式，建图需求深度数据格式
2. 学习单目地图初始化部分代码
3. 建立c++与python部分通信
4. 参考RGBD建图工程代码，学习建图思路

### 3/19实际工作
1. 调查了orbslam2使用的TUM格式数据集，MVSnet使用的replica数据集，确定好了输入和输出的数据集格式，并明确了如何进行计算和转化
2. 学习了部分单目地图初始化的代码

### 3/20计划工作

1. 学完单目地图初始化部分代码
2. 找RGBD建图工程代码，学习建图思路
3. 根据思路设计mvs部分代码，暂将输出定位深度图形式

### 3/20实际工作

1. 学完了单目初始化代码
2. 找到了常用的RGBD建图代码
3. 初步考虑slam部分思路

### 3/21计划工作

1. 学完地图点、关键帧、图结构、特征匹配部分代码
2. 阅读建图代码学习思路进一步考虑

### 3/21实际工作

1. 学完了地图点、关键帧、图结构的代码和特征匹配部分代码
2. 进一步设计slam部分代码

### 摆烂了两天

### 3/24-3/27工作

1. 简单看了跟踪、局部建图以及闭环检测和矫正代码
2. 设计mvsnet部分代码，暂定输出深度图像png和保存深度信息txt
3. 学习稠密建图代码思路

### 3/28-4/2计划工作

1. 简单看了BA优化、工程实践代码
2. 建立slam部分和mvsnet部分通信过程
3. 学会改进联合工程并编译，尝试同步估计、显示深度图和数值信息
4. 学习rgbd代码改造思路，并尝试修改代码

### 4/3工作

1. 将各部分连接完成毕业设计
2. 更新网络，优化效果