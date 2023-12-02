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
import FeatureSelection as FS
# 适用于噪声扰动的数据集

# 记录运行数据的表格所在目录
record_filepath = 'file_for_disambiguation_20230727/dataset_record.csv'

# 属性约简结果存放目录
reduction_result_file_path = 'file_for_disambiguation_20230727/result/'

# 使用的数据集
dataset_use_path = 'datasets_use.txt'

# 数据集所在文件夹目录
dataset_file_url = r'../../work4/001 20230727 Disambiguation using granular ball\disambiguation_result\noise_ratio_{}\{}.mat'

# 运行记录表的表头
record_head = ['数据集', '样本个数', '特征个数', '标签个数','噪声率' '数据预处理方式', 'param_for_radius', 'param_alpha', '约简后特征个数', '运行时间', '记录时刻', '运行设备']

# 进程数量
processing_num = 10

# 在该噪声比例下消歧的数据上进行特征选择
noise_ratio_list = [0.2, 0.4, 0.6, 0.8]

def main():
    # =======================  创建文件夹  ======================
    # 判断特征选择结果文件存放目录是否存在
    if not os.path.exists(reduction_result_file_path):
        os.makedirs(reduction_result_file_path)
    # 判断记录所使用数据集的文件是否存在
    if not os.path.exists(dataset_use_path):
        print('数据集 ' + dataset_use_path + ' 不存在')
        sys.exit()
    # 如果记录表csv文件不存在，则创建并写入表头
    if not os.path.exists(record_filepath):
        csv_file = csv.writer(open(record_filepath, 'w', newline='', encoding='utf_8_sig'))
        csv_file.writerow(record_head)
    # =========================================================

    # =======================  读取待使用数据集列表  ======================
    # dataset = ['CHD_49', 'Society', ]
    datasets = []
    f = open(dataset_use_path)
    line = f.readline()
    while line:
        # 读取没有被注释掉的数据集
        if line.find('//') == -1:
            datasets.append(line.replace('\n', ''))
        line = f.readline()
    f.close()
    print(datasets)
    # =======================参数设置==================================
    preProcessMethod = ['minMax']  # minMax, standard, mM_std, std_mM
    param_list_alpha = [0.01,0.1,1,5,10]
    k = [3,5,7,9,11]

    temp = list(itertools.product(datasets, noise_ratio_list, preProcessMethod, param_list_alpha, k))  # 笛卡尔积 进行参数排列组合 每个元组（数据集名称,参数1,参数2，...）
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
        preProcessMethod = params[2]
        param_alpha = params[3]
        param_k = params[4]

        share_lock.release()
        print("==== process {}, dataset:{}, noise:{},{}, alpha:{},k:{} time:{}".format(processing_index, dataset_name, noise_ratio, preProcessMethod,param_alpha,param_k,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # 防止在某个数据集上报错导致程序终止，使用try包裹
        try:
            # read a dataset
            data = scio.loadmat(dataset_file_url.format(noise_ratio, dataset_name))
            X = data['features'][:, :]
            Y = data['labels'][:, :]
            # normalize
            if preProcessMethod == 'minMax':
                # 将特征值进行归一化
                minMax = MinMaxScaler()  # Normalize data
                X = minMax.fit_transform(X[:, :])
            elif preProcessMethod == 'standard':
                # 将特征标准化
                Ss = StandardScaler()
                X = Ss.fit_transform(X[:, :])
            else:
                print("数据未经过预处理")
            features_rank, run_time = FS.FeatureSelection(X=X,Y=Y,param_alpha=param_alpha,param_k=param_k,dataset_name=dataset_name,noise_ratio=noise_ratio)
            # write the ranked features to txt file
            file_path = "{}/noise_ratio_{}/{}/{}/".format(reduction_result_file_path,noise_ratio, preProcessMethod, dataset_name)
            if os.path.exists(file_path) is False:
                os.makedirs(file_path)
            note = open("{}/{}_{}.txt".format(file_path, param_alpha,param_k), mode='w')
            note.write(','.join(str(i) for i in features_rank))
            note.close()
        except Exception as e:
            print(e)
            with open(record_filepath, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他锁
                writer = csv.writer(f)
                writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1],noise_ratio, preProcessMethod,str(param_alpha),str(param_k), 'Exception:' + str(e), '',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
                fcntl.flock(f, fcntl.LOCK_UN)
            continue
        # 将特征选择过程信息记录在记录表
        with open(record_filepath, 'a', newline='') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他锁
            writer = csv.writer(f)
            writer.writerow([dataset_name, X.shape[0], X.shape[1], Y.shape[1],noise_ratio, preProcessMethod,str(param_alpha),str(param_k), len(features_rank), run_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), socket.gethostname()])
            fcntl.flock(f, fcntl.LOCK_UN)
    share_lock.release()




if __name__ == '__main__':
    main()
