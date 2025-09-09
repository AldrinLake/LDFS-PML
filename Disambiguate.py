import os.path
import time


import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from GranularBall import GranularBall
from GranularBall import  GBList
from scipy.spatial.distance import cdist
import copy
import heapq

def Disambiguate(X,Y, ball_purity=1):
    start_time = time.time()
    # Data pre process
    instance_num, label_num = Y.shape
    Mm = MinMaxScaler()
    X_std = Mm.fit_transform(X[:, :])
    Y_confidence = copy.deepcopy(Y)
    Y_confidence = np.array(Y_confidence,dtype=float)
    # Disambiguation using label-specific information
    for label_index in range(label_num):
        # print('\r{}/{}'.format(label_index,label_num), end=' ')
        print('{}/{}'.format(label_index,label_num))
        data = np.hstack((X_std, np.array(Y[:,label_index]).reshape(instance_num, 1)))  # Add the label column to the last column of the data
        index = np.array(range(instance_num)).reshape(instance_num, 1)  # column of index
        data = np.hstack((data, index))  # Add the index column to the last column of the data
        # step 1: split ball
        granular_balls = GBList(data)  # create the list of granular balls
        print('start init ball')
        granular_balls.init_granular_balls(purity=ball_purity)  # initialize the list
        print('start remove negative balls')
        granular_balls.remove_negative_balls()
        print('start merge balls')
        granular_balls.merge_two_nearest_ball()
        granular_balls.remove_void_balls()
        # step 2: calculate label confidence for all positive labels
        ball_center_list = granular_balls.get_center()
        ball_size_list = granular_balls.get_data_size()
        # print(ball_size_list)
        # print(ball_size_list)
        positive_sample_index = np.where(Y[:, label_index] == 1)
        positive_data = data[positive_sample_index, :][0]

        # k=np.min([len(ball_size_list),5])
        # k_largset_ball_index = np.argpartition(np.array(ball_size_list), -k)[-k:]
        k=3
        if np.sum(np.array(ball_size_list)>=k)<=2:
            k=2
        if np.sum(np.array(ball_size_list)>=k)==0:
            k=1
        # print(k)
        # obtain data covered by balls which size are larger than k
        data_covered_by_ball_large_k = granular_balls.get_data_covered_by_ball_that_size_large_than_k(k=k)
        k_largset_ball_index = np.where(np.array(ball_size_list)>=k)
        for pd in positive_data:
            if pd.tolist() in data_covered_by_ball_large_k.tolist():
                continue
            matrix_similarity = get_similarity_matrix_use_gauss_kernel(X1=np.array([pd[:-2]]), X2=ball_center_list[k_largset_ball_index],sigma=1)
            try:
                similarity = np.max(matrix_similarity)
            except Exception as e:
                # print(e)
                continue
            Y_confidence[np.array(pd[-1],dtype=int),label_index] = similarity
    # row normalize for Y
    row_sum = np.sum(Y_confidence,axis=1)
    row_sum = np.where(row_sum==0,1,row_sum)  # guarantee the denominator is not zero
    for i in range(label_num):
        Y_confidence[:,i] = Y_confidence[:,i]/row_sum
    return Y_confidence, time.time() - start_time

def get_similarity_matrix_use_gauss_kernel(X1, X2, sigma):

    eul_distance = cdist(X1, X2, metric='euclidean')
    eul_distance2 = np.square(eul_distance)
    matrix_similarity = np.exp(- eul_distance2 / sigma)
    return matrix_similarity



if __name__ == '__main__':
    PML_data_path = r'E:\algorithm code\work3\000 ML data with noise labels\PML_datasets\{}\{}.mat'
    dataset_list = [
        # 'CHD_49',
        # 'Water-quality',
        # 'Flags',
        # 'Emotions',
        'VirusPseAAC',
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
        # 'Yeast'
    ]
    noise_ratio_list = [
        # 0.2,
        # 0.4,
        0.6,
        # 0.8
    ]
    for dataset in dataset_list:
        for noise_ratio in noise_ratio_list:
            print("start:{} with noise ratio of {}".format(dataset,noise_ratio))
            file_path = PML_data_path.format(noise_ratio, dataset)
            data = scio.loadmat(file_path)
            X = data['features'][:, :]
            Y_with_noise = data['labels'][:, :]
            Y_confidence = Disambiguate(X=X, Y=Y_with_noise)
            out_come_path = "disambiguation_result/noise_ratio_{}/".format(noise_ratio)
            if os.path.exists(out_come_path) == False:
                os.makedirs(out_come_path)
            scio.savemat("{}/{}.mat".format(out_come_path,dataset),{'features': X, 'labels': Y_confidence})