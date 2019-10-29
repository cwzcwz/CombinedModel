import os

path_dir = '../dataset/skeletondata'
path_dir_list = os.listdir(path_dir)
for i, path_name in enumerate(path_dir_list):
    print('path count:', i)
    file_path = os.path.join(path_dir, path_name)
    file_name_list = os.listdir(file_path)
    if len(file_name_list) != 1:
        print('skeleton path error:', file_path)
        break
    for file_name in os.listdir(file_path):
        file_name_path = os.path.join(file_path, file_name)
        for i, line in enumerate(open(file_name_path).readlines()):
            x = line.split(' ')
            # 用空格split后多了一列是"\n"
            if len(x) != 76:
                print(x[-2:])
                print('error file name:', file_name_path, len(x))
                print('error file lines:', i)
                break
print('is over')
