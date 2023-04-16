import os
import shutil
import pandas as pd

# 将此路径替换为您的CSV文件路径
csv_file_path = r'G:\456\4\4.10.csv'

# 将此路径替换为包含图片的文件夹路径
source_images_folder = r'G:\456\4'

# 将此路径替换为您希望将分类后的图片保存的文件夹路径
destination_folder = r'G:/new_train/456'

# 读取CSV文件
annotations = pd.read_csv(csv_file_path)

# 遍历CSV文件中的每一行
for index, row in annotations.iterrows():
    file_name = row['filename']
    expression = row['region_attributes']

    # 检查表情字段是否为空
    if not pd.isna(expression):
        # 获取表情值
        if eval(expression):
            expression_value = eval(expression)['expressions']
        else:
            expression_value = 3
        # 创建目标文件夹，如果不存在
        target_folder = os.path.join(destination_folder, f'expression_{expression_value}')
        os.makedirs(target_folder, exist_ok=True)

        # 将图片从源文件夹复制到目标文件夹
        shutil.copy(os.path.join(source_images_folder, file_name), os.path.join(target_folder, file_name))
