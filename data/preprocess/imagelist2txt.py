"""
将rgb或depth目录中的image按照时间顺序写入到txt文件中
"""

import os
from operator import itemgetter

depthimage_dir = 'E:/data/rgbimage'
# depthimage_dir = 'E:/data/depthimage'
image_path_list = os.listdir(depthimage_dir)

for image_path_name in image_path_list:
    image_path = os.path.join(depthimage_dir, image_path_name)

    txt = open(os.path.join('E:/data/rgbimage_list', image_path_name + '.txt'), 'w')
    # txt = open(os.path.join('E:/data/depthimage_list', image_path_name + '.txt'), 'w')

    image_list = os.listdir(image_path)

    image_list_splited = []
    for image_name in image_list:
        x_list = []
        a = image_name.split('-')
        a0 = a[0].split('.')
        a1 = a[1].split('.')
        a2 = a[2].split('.')
        x_list.append(int(a0[0]))
        x_list.append(int(a0[1]))
        x_list.append(int(a0[2]))
        x_list.append(int(a1[0]))
        x_list.append(int(a1[1]))
        x_list.append(int(a1[2]))
        x_list.append(int(a2[0]))
        image_list_splited.append(x_list)

    image_list_splited.sort(key=itemgetter(0, 1, 2, 3, 4, 5, 6))

    for i in image_list_splited:
        txt.write(str(i[0]) + '.' + str(i[1]) + '.' + str(
            i[2]) + '-' + str(i[3]) + '.' + str(i[4]) + '.' + str(i[5]) + '-' + str(i[6]) + '.png' + '\n')

    txt.close()
