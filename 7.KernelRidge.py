import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

rng = np.random.RandomState(0)
# print(rng)

X = 5 * rng.rand(100, 1)  # 随机化状态的数据不会变化

y = np.sin(X).ravel()  # 将二维数据变成一维数据
# print(X)
# print(np.sin(X))
# print(y)

# 给标签加噪声
# print(y[::5])  # 每隔五个数据取一个
# y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
# print(3 * (0.5-rng.rand(X.shape[0] // 5)))

y[::5] += 3 * (0.5 - rng.rand(20, 1).ravel())  # 每隔五个数据改变对应的值进去，范围在-1.5~1.5之间
# print(rng.rand(20, 1))

# 核邻回归
# kr = KernelRidge(kernel='sigmoid', alpha=0.3, gamma=0.3)
# kr = KernelRidge(kernel='linear', alpha=0.5, gamma=0.5)
# kr = KernelRidge(kernel='rbf', alpha=0.5, gamma=0.5)
# kr = KernelRidge(kernel='polynomial', alpha=0.5, gamma=0.5)
kr = GridSearchCV(KernelRidge(),  # 表格搜索（超级调参方法）
                  param_grid={"kernel": ['rbf', 'laplacian', 'polynomial', 'sigmoid'],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

# print(np.logspace(-2, 2, 5))  # [1.e-02 1.e-01 1.e+00 1.e+01 1.e+02]
kr.fit(X, y)
print(kr.best_score_, kr.best_params_)

X_plot = np.linspace(0, 5, 100)
# print(X_plot)

y_kr = kr.predict(X_plot[:, None])
# print(X_plot[:, None])  # 增加一个维度，变成形状为（100,1）的数据
# y_kr = kr.predict(np.expand_dims(X_plot, 1))
# print(y_kr)

plt.scatter(X, y)
plt.plot(X_plot, y_kr, color="red")
plt.show()



