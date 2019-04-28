# coding=utf-8
# 统计训练集的 mean 和　std 信息
# 标准化(Standardization)，或者去除均值和方差进行缩放
"""
将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


def test_algorithm():
    np.random.seed(123)
    print('use sklearn')
    # 注：shape of data: [n_samples, n_features]
    # Return a sample (or samples) from the "standard normal" distribution.
    # randn函数返回一个或一组样本，具有标准正态分布。
    data = np.random.randn(10, 4)
    scaler = StandardScaler()
    scaler.fit(data)
    trans_data = scaler.transform(data)
    print('original data: ')
    print(data)
    print('transformed data: ')
    print(trans_data)
    print('scaler info: scaler.mean_: {}, scaler.var_: {}'.format(scaler.mean_, scaler.var_))
    print('\n')

    print('use numpy by self')
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    var = std * std
    print('mean: {}, std: {}, var: {}'.format(mean, std, var))
    # numpy 的广播功能
    another_trans_data = data - mean
    # 注：是除以标准差
    another_trans_data = another_trans_data / std
    print('another_trans_data: ')
    print(another_trans_data)


if __name__ == '__main__':
    test_algorithm()
