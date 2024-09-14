# -*- coding: utf-8 -*-
"""
-------------------------------
    @软件：PyCharm
    @PyCharm：2023
    @Python：3.8
    @项目：HSI-UI
-------------------------------
    @文件：get_spectral.py
    @时间：2024/5/26 18:04
    @作者：XFK
    @邮箱：fkxing2000@163.com
# -------------------------------
"""
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np
import spectral.io.envi as envi
from spectral import spectral
import matplotlib.pyplot as plt
from spectral_pretreatment import spectral_pretreatment

spectral.settings.envi_support_nonlowercase_params = True


def mentionTheSpectrum(img, ScaleFactor):
    # 初始化
    height, width, num_bands = img.shape
    spectra = np.zeros((height, width, num_bands))
    # 提取每个像素的波谱线
    for i in range(height):
        for j in range(width):
            spectra[i, j] = img[i, j]
    # 创建布尔掩码，标记光谱值为0的像素点
    zero_mask = np.all(spectra == 0, axis=2)
    # 将光谱值为0的像素点置为NaN
    spectra[zero_mask] = np.nan
    spectra_2d = spectra.reshape(-1, num_bands)
    # 创建布尔掩码，仅包含有效像素点
    valid_mask = ~zero_mask.flatten()
    # 从有效的光谱中随机选择条数
    size = valid_mask.sum()
    select_times = 1
    for i in range(select_times):
        selected_indices = np.random.choice(np.where(valid_mask)[0], size=size, replace=False)
        selected_spectra = spectra_2d[selected_indices]
        # 计算平均光谱
        mean_selected_spectrum = np.mean(selected_spectra, axis=0)
        num_array = np.array(list(mean_selected_spectrum))
        normalized_array = np.divide(num_array, 255 / ScaleFactor)
        normalized_list = normalized_array.flatten()  # 将归一化后的数组转换为列表

    return normalized_list


def main(raw_file):
    hdr_file_path = raw_file[:-4] + '.hdr'
    B, G, R = [165, 106, 39]
    # 检查对应的hdr文件是否存在
    if not os.path.exists(hdr_file_path):
        print('文件异常，缺少对应的hdr文件')
    # 读入文件
    img = envi.open(hdr_file_path, raw_file)
    bands0 = img.load()
    bands = np.where(bands0 > 0.001, bands0, 0)
    ScaleFactor = bands.max()
    # 调整至0——255
    bands = (bands * (255 / bands.max())).astype('uint8')
    # 85-23
    gai = bands[:, :, 23] - bands[:, :, 85]

    # 进行二值化
    _, binary_image = cv.threshold(gai, 60, 255, cv.THRESH_BINARY)
    output = cv.connectedComponentsWithStats(binary_image, connectivity=8)
    labels, stats = output[1], output[2]
    largest_area_label = np.argmax(stats[1:, 4]) + 1
    mask = (labels == largest_area_label).astype(np.uint8) * 255
    mask0 = cv.bitwise_and(binary_image, binary_image, mask=mask)

    white_pixels = np.where(mask0 == 255)  # 获取白色像素点的坐标
    min_x, min_y = np.min(white_pixels, axis=1)  # 获取最小坐标
    max_x, max_y = np.max(white_pixels, axis=1)  # 获取最大坐标
    center_x = int((min_x + max_x) / 2)
    center_y = int((min_y + max_y) / 2)

    r = 120
    bandsed = bands  # [center_x - r:center_x + r, center_y - r:center_y + r]
    mask2 = mask0  # [center_x - r:center_x + r, center_y - r:center_y + r]
    height, width, num_bands = bandsed.shape
    result = np.zeros((height, width, num_bands), dtype=np.uint8)

    # 逐个提取波段并应用掩膜
    for i in range(num_bands):
        band = bandsed[:, :, i]
        masked_band = cv.bitwise_and(band, band, mask=mask2)
        result[:, :, i] = masked_band

    # 选择三个波段生成伪彩色图像
    rgb_raw = [bands[:, :, B], bands[:, :, G], bands[:, :, R]]
    rgb_bands = [result[:, :, B], result[:, :, G], result[:, :, R]]
    rgb_raws = cv.merge(rgb_raw)
    rgb_result = cv.merge(rgb_bands)

    # 绘制矩形框
    cv.rectangle(rgb_raws, (center_y - r, center_x - r), (center_y + r, center_x + r), (0, 255, 0), 1)
    output_spectral = mentionTheSpectrum(result, ScaleFactor)
    return rgb_raws, rgb_result, output_spectral


if __name__ == '__main__':
    raw_data = 'A203_RT.raw'
    rgb_raws, rgb_result, output_spectral = main(raw_data)
    # 截取数据
    output_spectral = output_spectral[15:237]
    feature_select = 'pollution'

    # 索引
    index_map = {
        'type': [33, 221, 220, 16, 168, 156, 164, 80, 125, 158],
        'jp': [217, 22, 146, 159, 186, 206, 38],
        'pollution': [220, 54, 86, 51, 211, 83, 144, 111, 140, 221, 186, 133, 106, 16, 175, 152, 206, 205, 165]
    }

    index = index_map[feature_select]

    if feature_select == 'jp':
        output_spectral = spectral_pretreatment(output_spectral, tool=2)
        output_spectraled = output_spectral[index]
    else:
        output_spectraled = output_spectral[index]

    plt.plot(output_spectral, label='Spectral'), plt.title('Spectral Data'), plt.xlabel('Wavelength'), plt.ylabel('Reflectance')
    plt.scatter(index, output_spectraled,
                edgecolors='red',
                facecolors='none',
                s=50, linewidth=2,
                label='Special Points',
                zorder=5),
    plt.legend(),
    plt.show()

    cv.imshow('gray_image1', rgb_raws)
    cv.imshow('gray_image2', rgb_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
