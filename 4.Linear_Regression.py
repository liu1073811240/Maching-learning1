from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# n_target表示回归一条线, random_state将数据固定
x, y = datasets.make_regression(n_samples=1000, n_features=1, n_targets=1, noise=50, random_state=0)

# 将数据和标签分成训练集、测试集（比例占0.3）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 创建回归模型
# reg = linear_model.LinearRegression()  # y=x, 线性回归
# reg = linear_model.Ridge(0, 5)  # l2.岭回归
# reg = linear_model.Lasso(0.1)  # L1，lasso回归
reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.2)  # 弹性网络：基于L1和L2的综合考虑
# reg = linear_model.LogisticRegression() # sigmoid, 逻辑斯蒂回归，（要求连续的数据）
# reg = linear_model.BayesianRidge()  # 贝叶斯回归

reg.fit(x_train, y_train)
# print(reg.coef_, reg.intercept_)  # 打印回归直线的斜率，截距

y_pred = reg.predict(x_test)  # 300张测试数据
# print(len(y_pred))

# 四种评价指标
# 1.平均绝对误差
# print(mean_absolute_error(y_test, y_pred))

# 2.均方误差
# print(mean_squared_error(y_test, y_pred))

# 3.回归模型的评价指标r2_score

print(r2_score(y_test, y_pred))

# 4.可解释性方差
# print(explained_variance_score(y_test, y_pred))

_x = np.array([-2.5, 2.5])
# print(_x.reshape(2, 1))
# print(np.expand_dims(_x, 1))
# print(_x[:, None])
# [[-2.5]
#  [ 2.5]]

x = _x.reshape(2, 1)

_y = reg.predict(x)
# print(_y)

plt.scatter(x_test, y_test)
# plt.scatter(x_test, y_pred)
plt.plot(x, _y, linewidth=3, color='orange')

plt.show()


