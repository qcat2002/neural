import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axe3d
from sklearn import preprocessing as pp
import pandas as pd


def read_data():
    # scikit learn (sklearn) -> preprocessing (normalisation)
    # pandas -> reading online datasets

    # csv online website
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # pandas reads csv
    # skiprows skips the irrelvant information 
    # header capture the name of each column
    # header = None -> automatically derives an integer column name 0,1,2,...
    raw_df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
    boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    x_input = boston_data[:, [2, 5]]
    y_output = raw_df.values[1::2, 2]
    return x_input, y_output


def dispay(x_input, y_output):
    # 使用其他支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 示例中使用 Arial Unicode MS 字体，您可以替换为其他中文支持字体
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x_input[:, 0], x_input[:, 1], y_output[:])

    # 设置轴标签
    ax.set_xlabel('第一维数据')
    ax.set_ylabel('第二维数据')
    ax.set_zlabel('输出参照')

    plt.show()


def linearmodel(w, b, x):
    return np.dot(x, w) + b


X, Y = read_data()
# dispay(X, Y)
