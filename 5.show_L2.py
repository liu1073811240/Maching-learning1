import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X 是10*10的希尔伯特矩阵
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# b = np.arange(0, 10)[:, np.newaxis]

y = np.ones(10)

print(X)
# print(b)

# 计算不同岭系数时的回归系数
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
# print(alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)  # 用列表装岭回归的斜率
    # print(coefs)

plt.rcParams['figure.figsize'] = (10, 6)  # 图像显示大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码，还有通过导入字体文件的方法
plt.rcParams['lines.linewidth'] = 0.5  # 设置

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim())

plt.xlabel('岭系数alpha')
plt.ylabel('回归系数coef')
plt.title('岭系数对回归系数的影响')

# plt.axis('tight')
plt.show()





