import os
from operator import itemgetter


def sorded_kinect_data(data_dir):
    txt_list = os.listdir(data_dir)
    txt_list_splited = []
    for txt_name in txt_list:
        x_list = []
        a = txt_name.split('-')
        a0 = a[0].split('.')
        a1 = a[1].split('.')
        a2 = a[2].split('.')
        x_list.append(int(a0[0]))  # 年
        x_list.append(int(a0[1]))  # 月
        x_list.append(int(a0[2]))  # 日
        x_list.append(int(a1[0]))  # 时
        x_list.append(int(a1[1]))  # 分
        x_list.append(int(a1[2]))  # 秒
        x_list.append(int(a2[0]))  # 毫秒
        x_list.append(a[3])
        x_list.append(a[4])
        txt_list_splited.append(x_list)

    txt_list_splited.sort(key=itemgetter(0, 1, 2, 3, 4, 5, 6))
    sorted_txt_list = []
    for i in txt_list_splited:
        sorted_txt_list.append(str(i[0]) + '.' + str(i[1]) + '.' + str(
            i[2]) + '-' + str(i[3]) + '.' + str(i[4]) + '.' + str(i[5]) + '-' + str(i[6]) + '-' + str(i[7]) + '-' + str(
            i[8]))
    return sorted_txt_list


def sorted_eeg_data(data_dir):
    txt_list = os.listdir(data_dir)
    txt_list_splited = []
    for txt_name in txt_list:
        x_list = []
        a = txt_name.split('_')
        # a2 = a[2]
        x_list.append(a[0])  # local
        x_list.append(a[1])  # 000000824
        # x_list.append(int(a2[0:4]))  # 年
        # x_list.append(int(a2[4:6]))  # 月
        # x_list.append(int(a2[6:8]))  # 日
        # x_list.append(int(a2[8:10]))  # 时
        # x_list.append(int(a2[10:12]))  # 分
        # x_list.append(int(a2[12:]))  # 秒
        x_list.append(int(a[2]))
        x_list.append(a[3])  # data
        x_list.append(a[4])  # 1/0
        txt_list_splited.append(x_list)
    txt_list_splited.sort(key=itemgetter(2))
    sorted_txt_list = []
    for i in txt_list_splited:
        sorted_txt_list.append(
            str(i[0] + '_' + str(i[1]) + '_' + str(i[2]) + '_' + str(i[3]) + '_' + str(i[4])))

    return sorted_txt_list
