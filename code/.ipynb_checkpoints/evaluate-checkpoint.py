'''
Author: your name
Date: 2022-03-21 11:19:30
LastEditTime: 2022-04-07 15:52:18
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/FM/evaluate.py
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from sklearn.cluster import KMeans
from model import DeepFM
from criteo import create_criteo_dataset, create_test_data
from utils import getFeature
import os

def get_test_data(x_file,y_file,test_column,log_path):
    test_x = np.loadtxt(x_file)
    test_y = np.loadtxt(y_file)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    test_x = pd.DataFrame(test_x,columns=features)
    test_y = pd.DataFrame(test_y,columns=['label'])

    test = pd.concat([test_x, test_y], axis=1)

    feat_value = test[test_column].value_counts()
    feat_value = feat_value.index
    feat_values = feat_value.values.astype('int32')
    test_data = []
    test_x_data = []
    test_y_data = []
    test_x = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    lenth = len(feat_values)
    for i in range(lenth):
        test_data.append(test.loc[test[test_column] == feat_values[i]])
        test_x_data.append(test_data[i][features].values.astype('int32'))  
        test_y_data.append(test_data[i]['label'].values.astype('int32')) 
        f = open(log_path, 'a')
        if i == 0:
            print('#'*50,file = f)
            print('sample num = {}'.format(len(test_y)),file = f)
        print('number of group {}: {}' .format(feat_values[i],len(test_x_data[i])),file = f)
        f.close()
    return (test_x, test_y), (test_x_data , test_y_data), feat_values

def get_model(model_path,feature_columns):
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]
    learning_rate = 0.001
    
    check_path = model_path
    model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    model.load_weights(check_path)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                        metrics=[AUC()])
    return model

def get_group_model(model_save_path,feat_value,feature_columns,version,test_column,model_name):
    lenth = len(feat_value)
    k = 8
    learning_rate = 0.0001
    model_list = []
    for i in range(lenth):
    # load model
        dnn_dropout = 0.5
        hidden_units = [256, 128, 64]
        learning_rate = 0.001

        model_tune_save_path = model_save_path + test_column + '/group' + str(feat_value[i]) + '/'
        check_path = model_tune_save_path + model_name + '-group' + str(feat_value[i]) + '-tune-' + str(version)+ 'w.ckpt'
        model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
        model.summary()
        # check_path = '/home/nesa320/huizhong/fm/saved_model/din_weights.val_loss_0.47201.ckpt'
        model.load_weights(check_path)

        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                        metrics=[AUC()])
        model_list.append(model)
    return model_list

def get_origin_auc_result(test_xy,test_xy_data,model,feat_values,log_path):
    batch_size = 4096
    test_x , test_y = test_xy
    test_x_data, test_y_data = test_xy_data
    lenth = len(feat_values)

    f = open(log_path, 'a')
    print('='*30,file = f)

    # AUC - all
    test_auc = model.evaluate(test_x, test_y, batch_size=batch_size)[1]
    print('Group=all AUC: %f' % test_auc,file = f)

    # AUC - every group
    for j in range(lenth):
        test_auc  = model.evaluate(test_x_data[j], test_y_data[j], batch_size=batch_size)[1]
        print('Group={} AUC: {}' .format(feat_values[j],test_auc),file = f)
    f.close() 

def get_auc(test_x_data,test_y_data,model,feat_values,log_path):
    batch_size = 4096

    f = open(log_path, 'a')
    print('='*30,file = f)

    # AUC - group
    test_auc = model.evaluate(test_x_data, test_y_data, batch_size=batch_size)[1]
    print('Group={} AUC: {}' .format(feat_values,test_auc),file = f)

