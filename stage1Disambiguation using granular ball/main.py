import os
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import csv
import sys
import socket
import fcntl
import multiprocessing
import numpy as np
import itertools

# 使用的数据集
from Disambiguate import Disambiguate

dataset_list = [
        # 'CHD_49',
        # 'Water-quality',
        # 'Flags',
        # 'Emotions',
        # 'VirusPseAAC',
        # 'Birds',
        # 'GpositivePseAAC',
        # 'PlantPseAAC',
        # 'GnegativePseAAC',
        # 'Image',
        # 'Scene',
        # '3sources_reuters1000',
        # '3sources_guardian1000',
        # '3sources_bbc1000',
        # 'CAL500',
        'Music_emotion',
        'Music_style',
        'Mirflickr',
    ]

# 数据集所在文件夹目录
# dataset_file_url = r'..\work3\000 ML data with noise labels\PML_datasets\{}\{}.mat'
dataset_file_url = r'..\..\PML datasets\{}\{}.mat'


# 进程数量
processing_num = 3

# 在该噪声比例下消歧的数据上进行标记消歧
noise_ratio_list = [
    'real',
    # 0.2,
    # 0.4,
    # 0.6,
    # 0.8
]

def main():
    # =======================参数设置==================================
    temp = list(itertools.product(dataset_list, noise_ratio_list))  # 笛卡尔积 进行参数排列组合 每个元组（数据集名称,参数1,参数2，...）
    temp.reverse()
    parameters_list = multiprocessing.Manager().list(temp)
    Lock = multiprocessing.Manager().Lock()
    proc_list = []
    for proc_index in range(processing_num):
        parameter_dist = {
            'proc_index': proc_index,
            'parameters_list': parameters_list,
            "lock": Lock
        }
        proc = multiprocessing.Process(target=SingleProcess,args=(parameter_dist,))
        proc_list.append(proc)
        proc.start()
        time.sleep(0.5)
    # 等待进程结束
    for proc in proc_list:
        proc.join()
    # 结束进程
    for proc in proc_list:
        proc.close()


def SingleProcess(parameter_dist):
    processing_index = parameter_dist['proc_index']
    share_lock = parameter_dist['lock']
    params_list = parameter_dist['parameters_list']
    while True:
        share_lock.acquire()
        if len(params_list) == 0:
            break
        params = params_list.pop()
        dataset_name = params[0]
        noise_ratio = params[1]

        share_lock.release()
        print("==== process {}, dataset:{}, noise:{}, time:{}".format(processing_index, dataset_name, noise_ratio,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # 防止在某个数据集上报错导致程序终止，使用try包裹
        try:
            # read a dataset
            data = scio.loadmat(dataset_file_url.format(noise_ratio,dataset_name))
            X = data['features'][:, :]
            Y_with_noise = data['labels'][:, :]
            Y_confidence, run_time = Disambiguate(X=X, Y=Y_with_noise)
            out_come_path = "disambiguation_result/noise_ratio_{}/".format(noise_ratio)
            if os.path.exists(out_come_path) == False:
                os.makedirs(out_come_path)
            scio.savemat("{}/{}.mat".format(out_come_path, dataset_name), {'features': X, 'labels': Y_confidence})

        except Exception as e:
            print(e)
            continue

    share_lock.release()




if __name__ == '__main__':
    main()
