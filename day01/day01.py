import numpy as np

# create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# print the array
print(arr)

# create a 2D array
arr2D = np.array([[1, 2, 3], [4, 5, 6]])

# print the 2D array
print(arr2D)

# create a 3D array
arr3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# print the 3D array        
print(arr3D)
    
print("Hello, world!")

import numpy as np
import matplotlib.pyplot as plt

# 创建单位正方形顶点（二维向量集合）
square = np.array([[0, 1, 1, 0, 0],  # x坐标
                   [0, 0, 1, 1, 0]]) # y坐标

# 定义复合线性变换：先旋转45°，再非均匀缩放
theta = np.pi / 4  # 45度弧度制
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
scaling = np.array([[1.5, 0],   # x轴缩放1.5倍
                    [0, 0.8]])  # y轴缩放0.8倍

# 矩阵乘法表示变换复合：先旋转后缩放
transform = scaling @ rotation  # 注意矩阵乘法顺序

# 应用变换：矩阵乘以每个向量
transformed = transform @ square

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始正方形
axes[0].plot(square[0], square[1], 'b-', linewidth=2)
axes[0].fill(square[0], square[1], 'blue', alpha=0.2)
axes[0].set_title('原始单位正方形', fontsize=14)
axes[0].set_xlabel('x轴')
axes[0].set_ylabel('y轴')
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# 变换后的平行四边形
axes[1].plot(transformed[0], transformed[1], 'r-', linewidth=2)
axes[1].fill(transformed[0], transformed[1], 'red', alpha=0.2)
axes[1].set_title('线性变换后：旋转45°+非均匀缩放', fontsize=14)
axes[1].set_xlabel('x轴')
axes[1].set_ylabel('y轴')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.show()

# 数学验证：变换矩阵的行列式值
det_value = np.linalg.det(transform)
print(f"变换矩阵的行列式值: {det_value:.4f}")
print(f"几何解释：面积变为原来的 {det_value:.2f} 倍")