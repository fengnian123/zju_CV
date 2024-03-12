## 1. 实验实现的功能简述及运行说明

### 1.1 实验实现功能

对文件夹中的图像进行相机标定和鸟瞰图转换操作，先进行相机标定得到相机的参数，再利用参数进行鸟瞰图转换

### 1.2 运行说明

- 程序运行：

  在命令行中输入: `path to folder\exe\main.exe ` 即可运行程序

  （例如D:\study\CV\3210102322\_Gaomingjian\exe\main.exe ）

- 运行的结果：

  输入图片集存放在`image`文件夹中

  角点检测后的图片存放在`image_with_corners`文件夹中

  鸟瞰图存放在`result`文件夹中

  

## 2.作业的开发与运行环境

- win11 系统 
- 使用python + Opencv库



## 3. 主要用到的函数与算法

### 3.1 Camera calibration

**Camera calibration**：相机标定是计算相机内参（如焦距、主点偏移）和畸变系数的过程，主要过程如下：

1. 定义世界坐标和图像坐标集合：通过选择适当的标定板（如棋盘格），将标定板的已知世界坐标系中的角点位置与对应的图像坐标系中检测到的角点位置进行匹配。假设点的世界坐标为 (X, Y, Z)，而像素坐标为 (u, v)，可以得到以下相机矩阵关系：

   <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231222100928.png" alt="image-20231231222100928" style="zoom:50%;" /> 

2. 进行亚像素角点检测：使用亚像素精度对图像中检测到的角点进行进一步的精炼，提高检测准确性。

3. 进行相机标定：传入世界坐标和图像坐标集合以及图像尺寸等参数计算相机的内参矩阵、畸变系数以及每幅图像的旋转向量和平移向量。

### 3.2 Bird's-Eye View Transform

**Bird's-Eye View Transform**：鸟瞰图变换是一种对图像进行透视投影变换的方法，用于将场景从俯视角度呈现。主要步骤如下：

1. 确定图像中的四个顶点位置。

2. 提供与这些顶点位置对应的目标输出图像中的四个顶点位置。

3. 根据这两组顶点位置，计算透视变换矩阵。

4. 使用计算得到的透视变换矩阵对输入图像进行变换，生成鸟瞰图结果。

   

### 3.3 其他函数

使用的其他库函数主要有：

- `retval, corners = cv2.findChessboardCorners(image, patternSize, flags=None)` 函数：在图像中找到棋盘格内角点

  参数如下：

  函数参数：

  - **image**: 输入的图像。
  - **patternSize**: 棋盘格内角点的行列数量
  - **flags**: 用于指定不同的标志来调整函数行为

  返回值：

  - **retval**: 指示棋盘格角点是否被成功检测到的布尔值。
  - **corners**: 存储检测到的角点位置的 numpy 数组。

- `corners = cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)`：用于对角点进行亚像素精化，参数如下：

  函数参数：

  - **image**: 输入的灰度图像。
  - **corners**: 检测到的角点位置。
  - **winSize**: 用于亚像素角点搜索的窗口大小。。
  - **zeroZone**: 这是搜索区域中心的压缩边界，表示在该区域内不进行搜索。
  - **criteria**: 表示终止迭代的条件。
  
  返回值：
  
  - **corners**: 经过亚像素调整后的角点位置。
  
- `retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix=None, distCoeffs=None, flags=None, criteria=None)`:通过角点来计算相机的内部参数矩阵和畸变系数

  函数参数：

  - **objectPoints**: 世界坐标系中的物体点
  - **imagePoints**: 对应于 `objectPoints` 的图像点，在图像上检测到的角点的坐标。
  - **imageSize**: 输入图像的大小。
  - **cameraMatrix**: 相机内部参数矩阵的初始估计值。
  - **distCoeffs**: 畸变系数的初始估计值。
  - **flags**: 标志参数，用于设定不同标定选项的组合。
  - **criteria**: 迭代的终止条件。

  返回值：

  - **retval**: 标定函数的运行状态指示符。
  - **cameraMatrix**: 得到的相机内部参数矩阵。
  - **distCoeffs**: 得到的畸变系数。
  - **rvecs**: 相机位置的旋转向量。
  - **tvecs**: 相机位置的平移向量。

- `dst = cv2.warpPerspective(src, M, dsize, flags=None, borderMode=None, borderValue=None)`: 用透视变换矩阵将图像从一个视角转换到另一个视角

  函数参数：

  - **src**: 输入图像。
  - **M**:透视变换矩阵.
  - **dsize**: 输出图像的大小。
  - **flags**: 插值方法的选择标志。
  - **borderMode**: 插值边界模式的选择。
  - **borderValue**: 当使用常量边界模式时，指定边界像素的值。

  返回值：

  - **dst**: 经过透视变换后的输出图像。



## 4. 实验步骤及代码具体实现
### 4.1 编写训练函数代码
* 将用到的 opencv 等库导入进来：
  
    ```python
    import os
    import cv2
    import numpy as np
    import glob
    ```



- 对图像进行角点检测和亚像素角点检测：

  - 遍历图像文件列表， `cv2.cvtColor()` 函数将图像转换为灰度图像调用 `cv2.findChessboardCorners()` 函数在灰度图像上检测棋盘格的内角点位置。

  - 在角点检测后，利用 `cv2.cornerSubPix()` 函数对角点进行亚像素角点检测。

  - 在原图上绘制角点后将图片保存下来

  ```python
  for idx, image_file in enumerate(image_files):
      img = cv2.imread(image_file)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      size = (gray.shape[1], gray.shape[0])
      # 角点检测
      ret, corners = cv2.findChessboardCorners(gray, (12, 12), None)
      # 亚像素角点检测
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      img_corners = img.copy()
      # 在原图上绘制角点
      cv2.drawChessboardCorners(img_corners, (12, 12), corners, ret)
      save_path = os.path.join('./image_with_corners', f'result_{idx}.jpg')
      cv2.imwrite(save_path, img_corners)
  ```



- 进行相机标定：

  - 存储棋盘格内角点对应的世界坐标。调用 `np.mgrid()`将其展开为一维数组，将展开后的网格坐标赋给 前两列构建棋盘格内角点对应的世界坐标。
  - 调用 `cv2.calibrateCamera()` 函数进行相机标定。
  - 得到返回的相机内部参数矩阵、畸变系数并打印保存
  
  ```python
  # 建立世界坐标系
      world_points = np.zeros((12 * 12, 3), np.float32)
      world_points[:, :2] = np.mgrid[0:12, 0:12].T.reshape(-1, 2)
      world_point_set = [world_points]
      image_point_set = [corners]
      # 相机标定
      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_point_set, image_point_set, size, None, None)
  ```
  
  

* 进行去畸变校正和鸟瞰图转换：

  - 使用 `cv2.undistort()` 函数对输入图像 `img` 进行去畸变校正

  - 从角点检测的结果 `corners` 数组中选择四个角点作为输入图像的顶点坐标，从之前定义的 世界坐标数组中选择对应的四个顶点表示鸟瞰图中区域的位置
  - 调用 `cv2.getPerspectiveTransform()` 函数计算透视变换矩阵 `M`
  - 接下来调用 `cv2.warpPerspective()` 函数对去畸变图像 `undistorted_img` 进行鸟瞰图转换并保存

  ```python
   # 鸟瞰图转换
      point1 = np.array([corners[0], corners[11], corners[-12], corners[-1]], dtype=np.float32)
      point2 = np.array([world_points[0, :2], world_points[11, :2], world_points[-12, :2], world_points[-1, :2]],
                        dtype=np.float32)
      point2 *= 60
      point2 += 400
      M = cv2.getPerspectiveTransform(point1, point2)
      out_img = cv2.warpPerspective(undistorted_img, M, size)
      save_path = os.path.join('./result', f'result_{idx}.jpg')
      cv2.imwrite(save_path, out_img)
  ```




### 4.2 结果测试

- 创建.spec配置文件，输入`pyinstaller -F main.py    `命令生成程序.spec配置文件和可执行文件，如下图所示：

    <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225202950073.png" alt="image-20231225202950073" style="zoom: 67%;" />  

    <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225203050604.png" alt="image-20231225203050604" style="zoom:67%;" />  

- 输入 `.\exe\main.exe` 运行生成出的训练程序可执行文件



## 5.  实验结果与分析

- 输入的棋盘图片：

  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230756100.png" alt="image-20231231230756100" style="zoom: 67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230816335.png" alt="image-20231231230816335" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230830207.png" alt="image-20231231230830207" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230849045.png" alt="image-20231231230849045" style="zoom:67%;" /> |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230905049.png" alt="image-20231231230905049" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230922603.png" alt="image-20231231230922603" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230936944.png" alt="image-20231231230936944" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231230948443.png" alt="image-20231231230948443" style="zoom:67%;" /> |

  

- 对齐后的部分图片（与上面对应）：

  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231019247.png" alt="image-20231231231019247" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231033409.png" alt="image-20231231231033409" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231053809.png" alt="image-20231231231053809" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231108948.png" alt="image-20231231231108948" style="zoom:67%;" /> |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231126770.png" alt="image-20231231231126770" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231138153.png" alt="image-20231231231138153" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231150476.png" alt="image-20231231231150476" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231214258.png" alt="image-20231231231214258" style="zoom:67%;" /> |
  
  
  
- 得到的鸟瞰图（与上面对应）：

  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231324727.png" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231353099.png" alt="image-20231231231353099" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231420083.png" alt="image-20231231231420083" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231437269.png" alt="image-20231231231437269" style="zoom:67%;" /> |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231455615.png" alt="image-20231231231455615" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231512547.png" alt="image-20231231231512547" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231534004.png" alt="image-20231231231534004" style="zoom:67%;" /> | <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231231231546962.png" alt="image-20231231231546962" style="zoom:67%;" /> |



## 6. 心得体会

在本次实验中，主要就是调用opencv库中的函数，所以参数的传入十分重要，在学习了使用的函数并调整参数后，完成了基本的目标



## 7. 参考文献

[1] [机标定（Camera calibration）原理、步骤](https://blog.csdn.net/lql0716/article/details/71973318)：https://blog.csdn.net/lql0716/article/details/71973318

[2] [Camera Calibration](https://blog.csdn.net/sdu_hao/article/details/104326116)：https://blog.csdn.net/sdu_hao/article/details/104326116
