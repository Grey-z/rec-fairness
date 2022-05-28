'''
Author: your name
Date: 2022-04-07 16:43:25
LastEditTime: 2022-04-07 17:15:52
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/deepfm/code/test.py
'''
'''
Author: your name
Date: 2022-04-07 10:29:06
LastEditTime: 2022-04-07 16:31:44
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/fm/cluster.py
'''
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from evaluate import *

def data_divide(num,data_x,data_y,cluster_label):
    x = []
    y = []
    for i in range(num):
        xi = []
        yi = []
        for j in range(len(cluster_label)):
            if cluster_label[j] == i:
                data_test = data_x[j].tolist()
                print(data_test)
                xi.append(data_x[j].tolist())
                yi.append(data_y[j].tolist())
        x.append(xi)
        y.append(yi)
    return x, y
def cluster_kmeans(num, data):
    data_cluster = KMeans(n_clusters=num, random_state=0).fit(data)
    return data_cluster.labels_  

names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
model_name = '/deepfm'

tune_data_file = '/home/nesa320/huizhong'  + model_name + '/dataset/test/tune.csv'

num = 2
sample_num = 10

tune = pd.read_csv(tune_data_file, sep=',', iterator=True, header=None,names = names)
tune = tune.get_chunk(sample_num)
tune = tune.reset_index(drop = True)
tune = tune.drop([0])
tune = tune.reset_index(drop = True)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
features = sparse_features + dense_features

tune_x = tune[features].values.astype('int32')
tune_y = tune['label'].values.astype('int32')

cluster_label = cluster_kmeans(num, tune_x)
test_x_data , test_y_data = data_divide(num,tune_x,tune_y,cluster_label)

print(test_x_data)