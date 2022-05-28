'''
Author: your name
Date: 2022-04-21 13:37:48
LastEditTime: 2022-04-21 13:42:10
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/deepfm/test_seg.py
'''
'''
Author: your name
Date: 2022-04-20 16:17:50
LastEditTime: 2022-04-20 19:18:48
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/fm/code/test_seg.py
'''
from segmen import *
from utils import *

test_x_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_x.npy'
test_y_file = '/home/nesa320/huizhong/deepfm/dataset/test/test_y.npy'
# log_path = '/home/nesa320/huizhong/deepfm/log/deepfm_1000w_622.txt'

model_origin_path = '/home/nesa320/huizhong/deepfm/save/deepfm-v1000-622.ckpt'

save_result_path_1 = '/home/nesa320/huizhong/deepfm/log/result_seg_421_metric1.npy'
save_result_path_2 = '/home/nesa320/huizhong/deepfm/log/result_seg_421_metric2.npy'
save_rule_path = '/home/nesa320/huizhong/deepfm/log/save_rule_421.txt'

sparse_seg_feature = ['C6','C9','C17','C20']
dense_seg_feature = []
distance = 10
theta = 0.01
num = 1000

#get pretrained model
feature_columns = getFeature(num)
model = get_model(model_origin_path,feature_columns)

# get all data
test = get_test_data(test_x_file,test_y_file)

test = test.sample(n=1000000)
#get all rule
seg_features = sparse_seg_feature + dense_seg_feature

all_rule = get_all_rule_dict(test,seg_features,distance)

save_result_1 = []
save_result_2 = []
save_rule = []
length_all = len(test)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
features = sparse_features + dense_features
test_x = test[features].values.astype('int32')
test_y = test['label'].values.astype('int32')
auc_all = get_auc(test_x,test_y,model)
for rule in all_rule:
    #get group satisfy / not satisfy rule
    select_data, unselect_data = get_evaluate_data(test,rule)
    select_data_x, select_data_y = select_data
    unselect_data_x, unselect_data_y = unselect_data

    length_group = len(select_data_x)
    select_rule = cut_rule(length_all, length_group, theta)
    if select_rule == 1:
        auc1 = get_auc(select_data_x,select_data_y,model)
        auc2 = get_auc(unselect_data_x,unselect_data_y,model)

        unfairness = auc2 - auc1
        unfairness2 = auc1 - auc_all
        save_result_1.append(unfairness)
        save_result_2.append(unfairness2)
        save_rule.append(rule)

np.savetxt(save_result_path_1,save_result_1,fmt='%.6f')
np.savetxt(save_result_path_2,save_result_2,fmt='%.6f')
np.savetxt(save_rule_path,save_rule,fmt='%s')