import numpy as np

'''
numpy.random.RandomState()是一个伪随机数生成器。那么伪随机数是什么呢？

伪随机数是用确定性的算法计算出来的是来自[0,1]均匀分布的随机数序列。
并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。
'''
a = np.random.RandomState(0)  # 将随机数状态固定下来
print(type(a), np.shape(a))  # <class 'mtrand.RandomState'>  ()
print(a.randn(1, 10))

b = np.random.randn(1, 1000)
print(b)
print(b.mean(), b.var())  





