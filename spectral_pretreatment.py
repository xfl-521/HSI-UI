# -*- coding: utf-8 -*-
"""
-------------------------------
    @软件：PyCharm
    @PyCharm：2023
    @Python：3.8
    @项目：HSI-UI
-------------------------------
    @文件：spectral_pretreatment.py
    @时间：2024/5/27 10:49
    @作者：XFK
    @邮箱：fkxing2000@163.com
# -------------------------------
"""
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def spectral_pretreatment(data, tool=None):
    """
    光谱预处理
    :param tool:
    :param data: 光谱数据
    :return: 预处理后的光谱数据
    """
    if tool == 0:
        # 数据sg平滑
        data = savgol_filter(data, window_length=5, polyorder=1)
    elif tool == 1:
        # 数据求一阶导
        data = np.gradient(data)
    elif tool == 2:
        # 数据求二阶导
        data = np.gradient(data)
        data = np.gradient(data)
    else:
        print('未选择正确的预处理工具')

    return data

if __name__ == '__main__':

    spectral_data = spectral_pretreatment(np.random.rand(100), tool=0)
    plt.plot(spectral_data), plt.show()
    print(spectral_data)