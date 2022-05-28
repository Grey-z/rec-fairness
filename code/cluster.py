'''
Author: your name
Date: 2022-04-07 10:29:06
LastEditTime: 2022-04-12 13:16:05
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/fm/cluster.py
'''
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from evaluate import *


def get_cluster_data(x_file,y_file,num,log_path):
    test_x = np.loadtxt(x_file,dtype=np.int32).tolist()
    test_y = np.loadtxt(y_file,dtype=np.int32).tolist()
    
    cluster_label = cluster_kmeans(num, test_x)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    test_x = pd.DataFrame(test_x,columns=features)
    test_y = pd.DataFrame(test_y,columns=['label'])
    data_label = pd.DataFrame(cluster_label,columns=['cluster'])

    test = pd.concat([test_x, test_y], axis=1)
    test = pd.concat([test,data_label],axis=1)

    test_data = []
    test_x_data = []
    test_y_data = []
    test_x = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')
    f = open(log_path, 'a')
    for i in range(num):
        test_data.append(test.loc[test['cluster'] == i])
        test_x_data.append(test_data[i][features].values.astype('int32'))  
        test_y_data.append(test_data[i]['label'].values.astype('int32')) 
        print('the number of group{} : {}'.format(i,len(test_x_data[i])), file=f)
    f.close()
    return (test_x, test_y), (test_x_data , test_y_data)

def get_sparse_cluster_data(x_file,y_file,num,log_path):
    test_x = np.loadtxt(x_file,dtype=np.int32).tolist()
    test_y = np.loadtxt(y_file,dtype=np.int32).tolist()

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    test_x = pd.DataFrame(test_x,columns=features)
    test_y = pd.DataFrame(test_y,columns=['label'])

    cluster_data = test_x[sparse_features].values.astype('int32')
    cluster_label = cluster_kmeans(num, cluster_data)
    
    data_label = pd.DataFrame(cluster_label,columns=['cluster'])

    test = pd.concat([test_x, test_y], axis=1)
    test = pd.concat([test,data_label],axis=1)

    test_data = []
    test_x_data = []
    test_y_data = []
    test_x = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')
    f = open(log_path, 'a')
    for i in range(num):
        test_data.append(test.loc[test['cluster'] == i])
        test_x_data.append(test_data[i][features].values.astype('int32'))  
        test_y_data.append(test_data[i]['label'].values.astype('int32')) 
        print('the number of group{} : {}'.format(i,len(test_x_data[i])), file=f)
    f.close()
    return (test_x, test_y), (test_x_data , test_y_data)

def cluster_kmeans(num, data):
    data_cluster = KMeans(n_clusters=num, random_state=0).fit(data)
    return data_cluster.labels_


test_x_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_x.npy'
test_y_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_y.npy'
log_path = '/home/nesa320/huizhong/deepfm/log/cluster_sparse_log.txt'
f = open(log_path, 'a')
model_origin_path = '/home/nesa320/huizhong/deepfm/save/deepfm-v1000-622.ckpt'

batch_size = 1000
sample_num = 1000 
feature_columns = getFeature(sample_num)

cluster_num = 4
xy, xy_data = get_sparse_cluster_data(test_x_file,test_y_file,cluster_num,log_path)

model = get_model(model_origin_path,feature_columns)

test_x, test_y = xy

test_x_data, test_y_data =xy_data

f = open(log_path, 'a')
test_auc  = model.evaluate(test_x, test_y, batch_size=batch_size)[1]
print('Group=all AUC: {}' .format(test_auc),file = f)

for j in range(cluster_num):
    test_auc  = model.evaluate(test_x_data[j], test_y_data[j], batch_size=batch_size)[1]
    print('Group={} AUC: {}' .format(j,test_auc),file = f)

f.close()