from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# 1.加载数据：内置数据（鸢尾花），鸢尾花数据集分类标签划分为3类，分别是山鸢尾（Iris-setosa）、变色鸢尾（Iris-versicolor）和维吉尼亚鸢尾（Iris-virginica）
iris = datasets.load_iris()
# print(iris)

# 2.拿到鸢尾花的数据和标签
x, y = iris.data, iris.target
# print(x.shape)  # (150, 4)  4表示一张图片4个特征，一张图片只有一个类别。
# print(y.shape)  # (150,)
# print(x)
# print(y)

# 3.根据拿到的数据和标签划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 将随机状态固定
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (105, 4) (105,) (45, 4) (45,)
'''
test_size：三种类型。float，int，None，可选参数。
            float：0.0-1.0之间。代表测试数据集占总数据集的比例。
            int：代表测试数据集具体的样本数量。
            None：设置为训练数据集。
            default：默认设置为0.25，当且train_size没有设置的时候，如果有设置，
                    则按照train_size的补来计算。

random_state：三种类型。int，randomstate instance，None。
        int：是随机数生成器的种子。每次分配的数据相同。
        randomstate：random_state是随机数生成器的种子。
        None：随机数生成器是使用了np.random的randomstate。
             种子相同，产生的随机数就相同。种子不同，即使是不同的实例，产生的种子也不相同
'''

# 4.数据预处理  (数据标准化处理：使用数据的均值和标准差缩放数据)
scaler = preprocessing.StandardScaler().fit(x_train)
print(x_train)
print(type(scaler))  # <class 'sklearn.preprocessing._data.StandardScaler'>

# 将标准化后的数据转成原有的numpy类型数据
x_train = scaler.transform(x_train)  # 使用训练数据的标准化参数μ，σ, 更具有代表性
x_test = scaler.transform(x_test)  # 使用训练数据的标准化参数μ，σ
print(x_train)
print(x_train.mean(), x_train.std())

# 5.创建KNN分类模型  （n_neighbors表示新数据圈的原数据量, 每个数据圈12个数据，再判断圈里面的数据属于哪个类别）
knn = neighbors.KNeighborsClassifier(n_neighbors=12)

# 6.模型拟合：给模型传入数据
knn.fit(x_train, y_train)  # 数据经过切割，需要进行拟合

# 7.选择交叉验证方式：数据分割比例，开始交叉训练验证
scores = cross_val_score(knn, x_train, y_train, cv=5, scoring="accuracy")  # cv：交叉验证生成器或可迭代的次数.(分为5组)
print(scores)  # 一共有五组的评分结果   [0.85714286 1.         1.         0.9047619  1.        ]
print(scores.mean())  # 0.9523809523809523

# 8.模型测试：使用测试集数据测试获得类别结果
y_pred = knn.predict(x_test)  # 将所测试的数据150*0.3=45张传入模型进行预测
print(y_pred)  # [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2 1 1 2 0 2 0 0]
print(y_test)  # [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 1 1 1 2 0 2 0 0]

# 测试集数据评估结果（精度）
print(accuracy_score(y_test, y_pred))  # 0.9777777777777777






