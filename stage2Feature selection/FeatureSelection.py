
import numpy as np
import scipy.io as scio
import torch
from BFSPML import BFSPML
from BFSPML_torch import BFSPML_torch
import time
from sklearn.preprocessing import MinMaxScaler
import PySimpleGUI as sg

def FeatureSelection(X,Y,param_alpha,param_k,dataset_name,noise_ratio):
    start_time = time.time()
    row_sum = np.sum(Y,axis=1)
    row_sum[row_sum==0]=1
    Y = Y / row_sum[:,np.newaxis]
    num, dim = X.shape
    # retain specific number of features according to the cardinality of feature set
    select_feature_number = None
    if dim > 1000:
        select_feature_number = int(dim * 0.1)
    if 1000 >= dim > 500:
        select_feature_number = int(dim * 0.2)
    if 500 >= dim > 100:
        select_feature_number = int(dim * 0.3)
    if dim <= 100:
        select_feature_number = int(dim * 0.4)
    select_feature_number = select_feature_number + 1

    candidate_features = list(range(dim))
    selected_features = []

    mc = BFSPML_torch(X=torch.from_numpy(X).type(torch.float32), Y=torch.from_numpy(Y).type(torch.float32), sigma=2, k=param_k)
    # mc = BFSPML(X=np.array(X,dtype=np.float32), Y=np.array(Y,dtype=np.float32), sigma=2, k=param_k)
    while True:
        flag = None
        min_value = float('inf')
        for f in candidate_features:
            selected_features.append(f)
            term_1, term_2 = mc.feature_score(feature_index=selected_features)
            # print("{},{}".format(term_1,term_2))
            f_score = term_1 +param_alpha*term_2
            if f_score<min_value:
                flag=f
                min_value =f_score
            selected_features.remove(f)
        # print("f:{},score:{}".format(flag,min_value))
        selected_features.append(flag)
        candidate_features.remove(flag)
        if len(selected_features) >= select_feature_number:
            break

        # 实时进度条
        sg.one_line_progress_meter('progress bar', len(selected_features), select_feature_number,
                                   "dataset:{} \nnoise ratio:{} \nparam_alpha:{}\n维度X{},Y{}\nselect:{}".format(
                                       dataset_name, noise_ratio, param_alpha, str(X.shape),
                                       str(Y.shape), str(flag)))

    end_time = time.time()
    # concat the ordered feature subset and unchecked feature subset [ordered feature subset，unchecked feature subset]
    final_result = selected_features + candidate_features
    # the initial number of the feature is set to 1 rather than 0
    final_result = list(np.array(final_result) + 1)
    return final_result, end_time - start_time

if __name__ == '__main__':
    datasetName = 'CHD_49'
    data = scio.loadmat('../../多标记数据集/' + datasetName + '.mat')
    X = data['features'][:, :]
    Y = data['labels'][:, :]
    # 将特征值进行归一化
    Ss = MinMaxScaler()  # Normalize data
    X = Ss.fit_transform(X[:, :])
    param_alpha = 0.5
    res, time = FeatureSelection(X=X,Y=Y, param_alpha=1,param_k=5,dataset_name=datasetName,noise_ratio=0)
    print(res)
    print(time)