from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score  # 引入交叉验证
import matplotlib.pyplot as plt

# 引入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x.shape)  # (150, 4)
print(y)

# 设置n_neighbors的值为1到30，通过绘图来看训练分数
k_range = range(1, 31)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(x, y) # 数据没有经过切割，可以不用再写fit()
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')  # 目的是分类
    k_score.append(scores.mean())

plt.figure()
plt.plot(k_range, k_score)
plt.xlabel("Value of k for KNN")
plt.ylabel("CrossValidation accuracy")
plt.show()
