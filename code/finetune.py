'''
Author: your name
Date: 2022-03-23 15:50:42
LastEditTime: 2022-04-05 15:13:15
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/fm/finetune.py
'''
'''
Author: your name
Date: 2022-03-21 11:19:30
LastEditTime: 2022-03-23 15:40:14
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/FM/evaluate.py
'''
import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import DeepFM
from criteo import create_criteo_dataset, create_test_data
from utils import getFeature


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# =============================== Parameters ===================================
model_name = '/deepfm'

tune_data_file = '/home/nesa320/huizhong'  + model_name + '/dataset/test/tune.csv'
# test_file = '/home/nesa320/huizhong/fm/dataset/criteo/test_data.txt'
log_path = '/home/nesa320/huizhong/deepfm/log/tune_log.txt'

model_origin_path = '/home/nesa320/huizhong/deepfm/save/deepfm-v1000-622.ckpt'
save_tune_path = '/home/nesa320/huizhong' + model_name + '/save_tune'

read_part = True

tune_version = [100,200]
feat_num = 1000
test_column = 'C20'

embed_dim = 8
dnn_dropout = 0.5
hidden_units = [256, 128, 64]

learning_rate = 0.0001
batch_size = 1000
epochs = 20

f = open(log_path , 'a')

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)
# # =============================== create test data ==============================
# test_x, test_y = create_test_data(file=test_file,
#                         read_part=False,
#                         sample_num=sample_num)
# feature_columns = getFeature()
# test = pd.concat([test_x, test_y], axis=1)

# test_data_1 = test.loc[ test['C9'] == 2]
# test_data_0 = test.loc[ test['C9'] == 0]

# sparse_features = ['C' + str(i) for i in range(1, 27)]
# dense_features = ['I' + str(i) for i in range(1, 14)]
# features = sparse_features + dense_features

# test_x = test[features].values.astype('int32')
# test_y = test['label'].values.astype('int32')

# test_x_1 = test_data_1[features].values.astype('int32')
# test_y_1 = test_data_1['label'].values.astype('int32')


# test_x_0 = test_data_0[features].values.astype('int32')
# test_y_0 = test_data_0['label'].values.astype('int32')

for version in tune_version:
    sample_num = version * 10000
    # =============================== create tune data ==============================
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
    tune = pd.read_csv(tune_data_file, sep=',', iterator=True, header=None,names = names)
    tune = tune.get_chunk(sample_num)
    tune = tune.reset_index(drop = True)
    tune = tune.drop([0])
    tune = tune.reset_index(drop = True)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    tune_x = tune[features].values
    tune_y = tune['label'].values.astype('int32')
    feature_columns = getFeature(feat_num)

    rowname = tune[test_column].value_counts()
    rownames = rowname.index
    row = rownames.values.astype('int32')

    tune_data = []
    tune_x_data = []
    tune_y_data = []

    lenth = len(row)
    for data_id in range(lenth):
        tune_data.append(tune.loc[tune[test_column] == row[data_id]])
        tune_x_data.append(tune_data[data_id][features].values.astype('int32'))  
        tune_y_data.append(tune_data[data_id]['label'].values.astype('int32')) 
        f = open(log_path, 'a')
        if data_id == 0:
            print('#'*50,file = f)
            print('sample num = {}'.format(sample_num),file = f)
        print('number of group {}: {}' .format(row[data_id],len(tune_x_data[data_id])),file = f)
        f.close()

    # ===============================  fine-tune ===================================
    
    for model_id in range(lenth):
        model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
        model.summary()

        model_tune_save_path = save_tune_path + '/' + test_column + '/group' + str(row[model_id])
        mkdir(model_tune_save_path)

        check_path = model_tune_save_path  + model_name  +'-group' + str(row[model_id]) + '-tune-' + str(version)+ 'w.ckpt'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss')
        model.load_weights(model_origin_path)

        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                        metrics=[AUC()])
        model.fit(
                tune_x_data[model_id],
                tune_y_data[model_id],
                epochs=epochs,
                callbacks=[checkpoint],
                batch_size=batch_size,
                validation_split=0.1
            )
