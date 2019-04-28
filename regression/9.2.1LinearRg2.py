import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pprint import pprint

if __name__ == "__main__":
    # 利用pandas的read_csv()方法读入数据集
    # 将其中的TV、Radio、Newspaper作为X,Sales作为Y
    data = pd.read_csv(u'Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # 实现在matplotlib中显示中文
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制关于TV、Radio、Newspaper和Sales的视图
    plt.figure(facecolor='w')
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    plt.xlabel('广告花费', fontsize=16)
    plt.ylabel('销售额', fontsize=16)
    plt.title('广告花费与销售额对比数据', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()

    # 调用sklearn.linear_model中的LinLinearRegression对象，创建实例
    linreg = LinearRegression()

    # 切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = linreg.fit(x_train, y_train)
    print("截距", linreg.intercept_)
    print("系数:", linreg.coef_)
    # [ 0.04458402  0.19649703 -0.00278146] 得到的系数是这个，
    # 就表示y = 2.99489303049533 + 0.04458402 * TV + 0.19649703 * Radio - 0.00278146 * Newspaper

    # 使用测试集进行预测x
    y_pred = linreg.predict(x_test)

    # 模型评估
    # 平均绝对误差MAE
    print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))

    # 均方误差MSE
    print("MSE: ", metrics.mean_squared_error(y_test, y_pred))

    # 均方根误差RMSE
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
