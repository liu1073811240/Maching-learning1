import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

rng = np.random.RandomState(0)
# print(rng)

X = 5 * rng.randn(100, 1)  # 随机化状态的数据不会变化
y1 = np.sin(X).ravel()  # 将二维数据变成一维数据

# y1 = ((1 - np.exp(-X)) / (1 + np.exp(-X))).ravel()

# y2 = np.cos(X).ravel()
y2 = 1 / (1 + np.exp(-X)).ravel()

# 给标签加噪声
# print(y[::5])  # 每隔五个数据取一个
# y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
# print(3 * (0.5-rng.rand(X.shape[0] // 5)))

y1[::5] += 3 * (0.5 - rng.rand(20, 1).ravel())  # 每隔五个数据加一个噪声进去
y2[::5] += 3 * (0.5 - rng.rand(20, 1).ravel())
# print(rng.rand(20, 1))

# 核邻回归
# kr = KernelRidge(kernel='sigmoid', alpha=0.3, gamma=0.3)
# kr = KernelRidge(kernel='linear', alpha=0.5, gamma=0.5)
# kr = KernelRidge(kernel='rbf', alpha=0.5, gamma=0.5)
# kr = KernelRidge(kernel='polynomial', alpha=0.5, gamma=0.5)
kr1 = GridSearchCV(KernelRidge(),  # 表格搜索（超级调参方法）
                  param_grid={"kernel": ['rbf', 'laplacian', 'polynomial', 'sigmoid'],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

kr2 = GridSearchCV(KernelRidge(),  # 表格搜索（超级调参方法）
                  param_grid={"kernel": ['rbf', 'laplacian', 'polynomial', 'sigmoid'],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

# print(np.logspace(-2, 2, 5))  # [1.e-02 1.e-01 1.e+00 1.e+01 1.e+02]
kr1.fit(X, y1)
kr2.fit(X, y2)
print(kr1.best_score_, kr1.best_params_)
print(kr2.best_score_, kr2.best_params_)

X_plot = np.linspace(-10, 10, 100)
# print(X_plot)

y1_kr = kr1.predict(X_plot[:, None])
y2_kr = kr2.predict(X_plot[:, None])
# print(X_plot[:, None])  # 增加一个维度，变成形状为（100,1）的数据
# y1_kr = kr1.predict(np.expand_dims(X_plot, 1))
# y2_kr = kr2.predict(np.expand_dims(X_plot, 1))
# print(y1_kr)

plt.scatter(X, y1)
plt.scatter(X, y2)
plt.plot(X_plot, y1_kr)
plt.plot(X_plot, y2_kr)
plt.show()



