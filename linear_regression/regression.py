import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axe3d
from sklearn import preprocessing as pp
import pandas as pd
import math


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
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x_input[:, 0], x_input[:, 1], y_output[:])

    # 设置轴标签
    ax.set_xlabel('第一维数据')
    ax.set_ylabel('第二维数据')
    ax.set_zlabel('输出参照')

    plt.savefig('boston_data.png')
    plt.show()


def linearmodel(xs, w, b):
    ps = np.zeros(np.shape(xs)[0])
    for i in range(np.shape(xs)[0]):
        ps[i] = np.dot(xs[i], w) + b
    return ps


def vectorized(xs, w, b):
    #                          (np matrix 1                  , m2) tuple
    xs = np.concatenate((np.ones((np.shape(xs)[0], 1)), xs), axis=1)
    w = np.concatenate((np.array([b]), w))
    return xs, w


def vectorized_linearmodel(xs_vec, w_vec):
    return np.dot(xs_vec, w_vec)


def solve_exact(X, y):
    A = np.dot(X.T, X)
    c = np.dot(X.T, y)
    return np.dot(np.linalg.inv(A), c)


def cost(prediction, target):
    total_residual = np.sum((prediction - target))
    return np.dot(total_residual, total_residual)/(2*prediction.shape[0])

