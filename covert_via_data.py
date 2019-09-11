#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import cv2
import numpy as np
import math


csv_data = pd.read_csv('data/qrqm/train/qrqm05Sep2019_19h34m28s_export.csv',usecols=[1,4])  # 读取训练数据
# print(csv_data.shape)  # (610, 6)

src_img_dir_path = '/home/xyb/workspace/unet/data/qrqm/train/image'
tar_img_dir_path = '/home/xyb/workspace/unet/data/qrqm/train/label'

point_color = (0, 0, 0) # BGR
thickness = 10
lineType = 4

# print(csv_data)

before_img = ''
img_points_dict = {'test':1}
for row in csv_data.itertuples():
    print(row.file_list,row.spatial_coordinates)
    img = row.file_list[2:-2]
    points = row.spatial_coordinates[3:-1].split(',')
    if img in img_points_dict: #处理过该图片
        ptStart = (math.ceil(float(points[0])), math.ceil(float(points[1])))
        ptEnd = (math.ceil(float(points[2])), math.ceil(float(points[3])))
        cv2.line(tar_img, ptStart, ptEnd, point_color, thickness, lineType)
        tar_img_path = os.path.join(tar_img_dir_path, img)
        cv2.imwrite(tar_img_path, tar_img)
    else:#没有处理过该图片,创建该图片一样大小打白色图
        src_img_path = os.path.join(src_img_dir_path, img)
        im = cv2.imread(src_img_path)
        tar_img = np.ones(im.shape, np.uint8)*255
        ptStart = (math.ceil(float(points[0])),math.ceil(float(points[1])))
        ptEnd = (math.ceil(float(points[2])), math.ceil(float(points[3])))
        cv2.line(tar_img, ptStart, ptEnd, point_color, thickness, lineType)
        img_points_dict[img] = 'draw'
        before_img = img
        tar_img_path = os.path.join(tar_img_dir_path, img)
        cv2.imwrite(tar_img_path,tar_img)
