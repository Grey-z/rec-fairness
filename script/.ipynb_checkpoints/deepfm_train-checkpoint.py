import pandas
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm
from ast import literal_eval

deepctr_path = '/root/linghui/rec-fairness/'
sys.path.append(deepctr_path)
import deepctr
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
# from utilis import get_feat_dict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def get_feat_dict(data):
    if data == 'ml-1m':
        feat_dict = {'movie_id': 3706,
                 'user_id': 6040,
                 'gender': 2,
                 'age': 7,
                 'occupation': 21,
                 'zipcode': 3439}
    return feat_dict

#path 
saved_model_path = '/data/linghui/saved_model/deepfm-ml-1m/'
saved_data_path = '/data/linghui/ml-1m/processed_data/'
train_data_path = saved_data_path + 'train_data.csv' 
test_data_path = saved_data_path + 'test_data.csv'

#params
movie_size = 18
max_len = 6

#load data
train_data = pd.read_csv(train_data_path,index_col=0)

genres_list = train_data['genres'].tolist()
genres = []
for i in range(len(genres_list)):
    genres.append(literal_eval(genres_list[i]))
train_data['genres'] = genres

#get model initial feats
dataset = 'ml-1m'
feat_dict = get_feat_dict(dataset)

sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zipcode"]

selected_feature = ["movie_id", "user_id"]
fixlen_feature_columns = [SparseFeat(feat, feat_dict[feat], embedding_dim=4)
								  for feat in selected_feature]

# use_weighted_sequence = False
# if use_weighted_sequence:
#     varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size = movie_size + 1, 
#                                                           embedding_dim=4), maxlen=max_len, combiner='mean', weight_name='genres_weight')]  
#     # Notice : value 0 is for padding for sequence input feature
# else:
#     varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size = movie_size + 1, 
#                                                           embedding_dim=4), maxlen=max_len, combiner='mean', weight_name=None)]  
#     # Notice : value 0 is for padding for sequence input feature

# linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
# dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

linear_feature_columns = fixlen_feature_columns 
dnn_feature_columns = fixlen_feature_columns 

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train_model_input = {name: train_data[name].values for name in sparse_features}
label = train_data['rating'].values

#build model
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['AUC', 'Precision', 'Recall'])
model.summary()

batch_size = 256
epochs = 20

# check_path = saved_model_path + 'deepfm-ml-1m.ckpt'
check_path = '/data/linghui/saved_model/deepfm-ml-1m/deepfm-no-sfeat/' + 'deepfm-ml-1m-del-sf.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 )
print('begain training')
model.fit(train_model_input, label,
						batch_size=batch_size, epochs=epochs, verbose=2,
                        callbacks=[checkpoint],
						validation_split=0.2)