#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import sklearn.datasets
from pprint import pprint
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings


def not_empty(s):
    return s != ''


if __name__ == "__main__":
    # 忽略warning的输出
    warnings.filterwarnings(action='ignore')
    """
    If True, always print floating point numbers using fixed point notation, 
    in which case numbers equal to zero in the current precision will print as zero. 
    If False, then scientific notation is used when absolute value of the smallest number is < 1e-4 or the ratio of 
    the maximum absolute value to the minimum is > 1e3. The default is False.
    True的话，会输出浮点，否则科学计数法，默认科学计数法
    """
    np.set_printoptions(suppress=True)
    file_data = pd.read_csv('./housing.data', header=None)
    # a = np.array([float(s) for s in str if s != ''])
    """
    np.empty()创建多维数组
    """
    data = np.empty((len(file_data), 14))
    for i, d in enumerate(file_data.values):
        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
        data[i] = d
    x, y = np.split(data, (13,), axis=1)
    # data = sklearn.datasets.load_boston()
    # x = np.array(data.data)
    # y = np.array(data.target)
    print('样本个数：%d, 特征个数：%d' % x.shape)
    print(y.shape)
    """
    y = [[]]
    after ravel()
    y = []
    """
    y = y.ravel()
    """
    sklearn.model_selection.train_test_split()
    随机划分训练集和测试集
    test_size：样本占比，如果是整数的话就是样本的数量
    random_state：是随机数的种子。
    随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，
    其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
    随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
    种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
    
    Returns:	splitting : list, length=2 * len(arrays)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    # model = Pipeline([
    #     ('ss', StandardScaler()),
    #     ('poly', PolynomialFeatures(degree=3, include_bias=True)),
    #     ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
    #                             fit_intercept=False, max_iter=1e3, cv=3))
    # ])
    """
    随机森林 的 模型
    
    n_estimators : integer, optional (default=10)
    The number of trees in the forest.
    Changed in version 0.20: 
    The default value of n_estimators will change from 10 in version 0.20 to 100 in version 0.22.
    
    criterion : string, optional (default=”mse”)
    The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, 
    which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.
    """
    model = RandomForestRegressor(n_estimators=50, criterion='mse')
    print('开始建模...')
    model.fit(x_train, y_train)
    # linear = model.get_params('linear')['linear']
    # print u'超参数：', linear.alpha_
    # print u'L1 ratio：', linear.l1_ratio_
    # print u'系数：', linear.coef_.ravel()

    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('均方误差：', mse)

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
    plt.legend(loc='best')
    plt.title('波士顿房价预测', fontsize=18)
    plt.xlabel('样本编号', fontsize=15)
    plt.ylabel('房屋价格', fontsize=15)
    plt.grid()
    plt.show()
