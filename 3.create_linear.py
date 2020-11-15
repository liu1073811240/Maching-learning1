# 使用SKlearn 创建线性回归数据

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# n_samples：生成样本数1000   n_feature:每个样本一个特征数     noise:样本随机噪点数     coef:是否返回回归系数
x, y, coef = make_regression(n_samples=1000, n_features=1, noise=50, coef=True)
print(len(x))
print(len(y))
print(coef)  # 返回底层线性模型的系数，类似斜率

# 画图
plt.scatter(x, y)
plt.plot(x, x*coef, color="red", linewidth=2)
plt.show()








