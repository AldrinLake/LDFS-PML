import numpy as np

# 示例数组
arr = np.array([[4,1, 2, 3]])

# 给定值
threshold = 3

# 找到小于给定值的元素的坐标
indices = np.where(arr < threshold)

# 打印结果
print("Indices of elements less than", threshold, ":")
print(indices[1])
