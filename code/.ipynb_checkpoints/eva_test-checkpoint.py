'''
Author: your name
Date: 2022-04-06 14:18:56
LastEditTime: 2022-04-06 14:46:06
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/deepfm/eva_test.py
'''
from evaluate import *

test_x_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_x.npy'
test_y_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_y.npy'
log_path = '/home/nesa320/huizhong/deepfm/log/deepfm_1000w_622.txt'
test_column = 'C20'

xy , xy_data , feat_value = get_test_data(test_x_file,test_y_file,test_column,log_path)

sample_num = 1000 
feature_columns = getFeature(sample_num)

model_save_path = '/home/nesa320/huizhong/deepfm/save_tune/'
version = 100
model_name = 'deepfm'
model_list = get_group_model(model_save_path,feat_value,feature_columns,version,test_column,model_name)

for i in range(len(feat_value)):
    print('')
    x_data, y_data = xy_data
    get_auc(x_data[i],y_data[i],model_list[i],feat_value[i],log_path)