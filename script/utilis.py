import sys
import tensorflow as tf
deepctr_path = '/root/linghui/rec-fairness/'
sys.path.append(deepctr_path)
import deepctr
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names

def get_feat_dict(data):
    if data == 'ml-1m':
        feat_dict = {'movie_id': 3706,
                 'user_id': 6040,
                 'gender': 2,
                 'age': 7,
                 'occupation': 21,
                 'zipcode': 3439}
    return feat_dict

def get_saved_model_details(model, cate):
    if model == 'DeepFM':
        if cate == 'all-feat':
            check_path = '/data/linghui/saved_model/deepfm-ml-1m/deepfm-all-feature/deepfm-ml-1m.ckpt'
            selected_feat = ["movie_id", "user_id", "gender", "age", "occupation", "zipcode"]
        if cate == 'del-sf':
            check_path = '/data/linghui/saved_model/deepfm-ml-1m/deepfm-del-sf/deepfm-ml-1m-del-sf.ckpt'
            selected_feat = ["movie_id", "user_id"]
    return check_path , selected_feat
            
            
            

def get_model(model_params,data_params):
    dataset = data_params['dataset']

    model_name = model_params['model_name']
    cate = model_params['cate']
    embedding_dim = model_params['embedding_dim']
    check_path, selected_feature = get_saved_model_details(model_name,cate)
    
    feat_dict = get_feat_dict(dataset)
    fixlen_feature_columns = [SparseFeat(feat, feat_dict[feat], embedding_dim=embedding_dim)
                                      for feat in selected_feature]
    linear_feature_columns = fixlen_feature_columns 
    dnn_feature_columns = fixlen_feature_columns 
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    

    if model_name == 'DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.summary()
    model.load_weights(check_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
                  metrics=['AUC', 'Precision', 'Recall'])
    return model



'''
Author: your name
Date: 2022-04-21 13:36:03
LastEditTime: 2022-04-21 13:38:10
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/deepfm/code/segmen.py
'''
'''
Author: your name
Date: 2022-04-20 15:33:52
LastEditTime: 2022-04-20 18:56:08
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/fm/code/segmen.py
'''
import itertools
from unittest import result

import tensorflow as tf
import pandas as pd
import math
import numpy as np

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import DeepFM
from criteo import create_criteo_dataset, create_test_data
from utils import *

def create_single_rule(key,value):
    return{key:value}

def get_single_sparse_rule(df,features):
    single_rule = []
    feat_value = df[features].value_counts()
    feat_value = feat_value.index
    sparse_feat_value = feat_value.values.astype('int32')
    for value in sparse_feat_value:
        rule = create_single_rule(features,value)
        single_rule.append(rule)
    return single_rule

def get_single_dense_rule(df,feature,k):
    min_value = df[feature].min()
    max_value = df[feature].max()
    single_rule = []
    d = math.ceil((max_value - min_value) / k)
    for i in range(k):
        x = min_value + i * d
        y = x + d
        single_rule.append({feature:[x,y]})
    return single_rule

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def combine_rule(rule_dict_1,rule_dict_2):
    len1 = len(rule_dict_1)
    len2 = len(rule_dict_2)
    all_rule = []
    for i in range(len1):
        rule1 = rule_dict_1[i]
        for j in range(len2):
            rule2 = rule_dict_2[j]
            merge_rule = merge_two_dicts(rule1,rule2)
            all_rule.append(merge_rule)
    return all_rule

def creat_rule_query(rule):
    result = ''
    key = rule.keys()
    length = len(key)
    count = 0
    for i in key:
        count = count + 1
        if i[0] == 'C':
            result = result + '( ' + i +' == ' + str(rule[i]) + ')'
        elif i[0] == 'I':
            result = result + '(' + str(rule[i][0]) + ' <= ' + i + ' <= ' +  str(rule[i][1]) + ')'
        if count < length:
            result = result + ' & '
    return result

def get_all_rule_dict(data,seg_features,k):
    single_rule_dict = []
    all_rule_dict = []
    for feature in seg_features:
        temp_rule = []
        if feature[0] == 'C':
            single_rule = get_single_sparse_rule(data,feature)
        if feature[0] == 'I':
            single_rule = get_single_dense_rule(data,feature,k)
        
        single_rule_dict = single_rule_dict + single_rule
        if len(all_rule_dict) > 0: 
            temp_rule = combine_rule(all_rule_dict,single_rule)
        all_rule_dict = all_rule_dict + single_rule + temp_rule
    return all_rule_dict

def get_evaluate_data(data,rule):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    result = creat_rule_query(rule)
    select_data = data.query(result)
    unselect_data = data.drop(select_data.index)

    select_data_x = select_data[features].values.astype('int32')
    select_data_y = select_data['label'].values.astype('int32')
    unselect_data_x = unselect_data[features].values.astype('int32')
    unselect_data_y = unselect_data['label'].values.astype('int32')

    return (select_data_x,select_data_y),(unselect_data_x,unselect_data_y)

def cut_rule(length_all,length_group,theta):
    x = length_group / length_all
    if (x >= theta) & (x < 1-theta):
        return 1
    else:
        return 0

def get_test_data(x_file,y_file):
    test_x = np.loadtxt(x_file)
    test_y = np.loadtxt(y_file)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    test_x = pd.DataFrame(test_x,columns=features)
    test_y = pd.DataFrame(test_y,columns=['label'])

    test = pd.concat([test_x, test_y], axis=1)
    return test

def get_auc(test_x,test_y,model):
    batch_size = 1000
    test_auc = model.evaluate(test_x, test_y, batch_size=batch_size)[1]
    return test_auc

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


