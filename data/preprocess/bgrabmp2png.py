"""
将bgra图像转换为png图像方法
"""

import os
import cv2

bmp_brga_path_dir = 'E:\\data1\\bgraimage'
bmp_new_path_dir = 'E:\\data\\rgbimage'

path_dir_list = os.listdir(bmp_brga_path_dir)
for path_dir in path_dir_list:
    file_dir_path = os.path.join(bmp_brga_path_dir, path_dir)
    print(file_dir_path)
    if file_dir_path not in ['E:\\data1\\bgraimage\\2019.6.18-18.3.49-83-v1',
                             'E:\\data1\\bgraimage\\2019.6.18-18.26.23-438-v1']:
        continue
    file_new_dir_path = os.path.join(bmp_new_path_dir, path_dir)
    os.makedirs(file_new_dir_path)
    for file_name in os.listdir(file_dir_path):
        print(file_name)
        file_name_path = os.path.join(file_dir_path, file_name)
        image = cv2.imread(file_name_path)
        file_name_new_path = os.path.join(file_new_dir_path, str(file_name.split('bmp')[0]) + 'png')
        cv2.imwrite(file_name_new_path, image)
