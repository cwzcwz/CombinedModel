"""
将depth深度图像转换为png图像
"""

import os
import cv2

bmp_brga_path_dir = 'E:\\data2\\depthimage'
bmp_new_path_dir = 'E:\\data\\depthimage'

path_dir_list = os.listdir(bmp_brga_path_dir)
for path_dir in path_dir_list:
    file_new_dir_path = os.path.join(bmp_new_path_dir, path_dir)
    os.makedirs(file_new_dir_path)
    file_dir_path = os.path.join(bmp_brga_path_dir, path_dir)
    for file_name in os.listdir(file_dir_path):
        print(file_name)
        file_name_path = os.path.join(file_dir_path, file_name)
        image = cv2.imread(file_name_path, cv2.IMREAD_GRAYSCALE)
        print(file_name.split('bmp')[0])
        file_name_new_path = os.path.join(file_new_dir_path, str(file_name.split('bmp')[0]) + 'png')
        cv2.imwrite(file_name_new_path, image)
