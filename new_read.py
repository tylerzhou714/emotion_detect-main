import os
import shutil
import cv2
import pandas as pd
import numpy as np

def crop_face(image_path, face_coordinates):
    image = cv2.imread(image_path)
    x, y, w, h = face_coordinates
    face_crop = image[y:y+h, x:x+w]
    return face_crop

csv_file_path = r'G:\456\6\4.10.csv'
source_images_folder = r'G:\456\6'
destination_folder = r'G:/new_train/456'

annotations = pd.read_csv(csv_file_path)

# 打乱数据
annotations = annotations.sample(frac=1).reset_index(drop=True)

num_samples = len(annotations)
train_ratio = 0.8

train_index = int(num_samples * train_ratio)

for index, row in annotations.iterrows():
    file_name = row['filename']
    expression = row['region_attributes']

    # 解析 region_shape_attributes 列以获取人脸坐标
    region_shape_attributes = eval(row['region_shape_attributes'])
    if region_shape_attributes:
        all_points_x = region_shape_attributes['all_points_x']
        all_points_y = region_shape_attributes['all_points_y']
        x, y, w, h = min(all_points_x), min(all_points_y), max(all_points_x) - min(all_points_x), max(all_points_y) - min(all_points_y)
    else:
        continue

    # 检查表情字段是否为空
    if not pd.isna(expression):
        if eval(expression):
            expression_value = eval(expression)['expressions']
        else:
            expression_value = 3

        # 根据索引确定数据集（训练、验证或测试）
        if index < train_index:
            dataset = 'train'
        else :
            dataset = 'val'

        target_folder = os.path.join(destination_folder, dataset, f'{expression_value}')
        os.makedirs(target_folder, exist_ok=True)

        image_path = os.path.join(source_images_folder, file_name)
        cropped_face = crop_face(image_path, (x, y, w, h))
        cropped_image_path = os.path.join(target_folder, file_name)
        cv2.imwrite(cropped_image_path, cropped_face)
