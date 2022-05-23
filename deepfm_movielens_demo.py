import pandas
import pandas as pd
import sklearn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm

from deepctr.models import DeepFEFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
import numpy as np

def neg_sample(u_data, neg_rate=1):
    # 全局随机采样
    item_ids = u_data['movie_id'].unique()
    print('start neg sample')
    neg_data = []
    # 负采样
    for user_id, hist in tqdm(u_data.groupby('user_id')):
        # 当前用户movie
        rated_movie_list = hist['movie_id'].tolist()
        candidate_set = list(set(item_ids) - set(rated_movie_list))
        neg_list_id = np.random.choice(candidate_set, size=len(rated_movie_list) * neg_rate, replace=True)
        for id in neg_list_id:
            neg_data.append([user_id, id, -1, 0])
    u_data_neg = pd.DataFrame(neg_data)
    u_data_neg.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    u_data = pd.concat([u_data, u_data_neg])
    print('end neg sample')
    return u_data

#读取rating信息
u_data_file = '/home/nesa320/huizhong/movielens/data-ml/ml-100k/u.data'
u_data = pd.read_csv(u_data_file, sep='\t', header=None)
u_data.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
#负采样
u_data = neg_sample(u_data, neg_rate=1)

#读取movie数据
item_file = '/home/nesa320/huizhong/movielens/data-ml/ml-100k/u.item'
u_item = pd.read_csv(item_file, sep='|', header=None, error_bad_lines=False,engine='python', encoding="ISO-8859-1")
genres_columns = ['Action', 'Adventure',
                    'Animation',
                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western']

u_item.columns = ['movie_id', 'title', 'release_date', 'video_date', 'url', 'unknown'] + genres_columns

#处理generes
genres_list = []
for index, row in u_item.iterrows():
    genres = []
    for item in genres_columns:
        if row[item]:
            genres.append(item)
    genres_list.append('|'.join(genres))

u_item['genres'] = genres_list
for item in genres_columns:
    del u_item[item]

def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))

#读取用户信息
user_file = '/home/nesa320/huizhong/movielens/data-ml/ml-100k/u.user'
u_user = pd.read_csv(user_file, sep='|', header=None)
u_user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip']

data = pd.merge(u_data, u_item, on="movie_id", how='left')
data = pd.merge(data, u_user, on="user_id", how='left')
# data.to_csv('ml-100k/data.csv', index=False)

#分离特征
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]

data[sparse_features] = data[sparse_features].astype(str)
target = ['rating']

# 评分
data['rating'] = [1 if int(x) >= 0 else 0 for x in data['rating']]
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
								  for feat in sparse_features]

use_weighted_sequence = False
if use_weighted_sequence:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
else:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

data = sklearn.utils.shuffle(data)
train_model_input = {name: data[name].values for name in sparse_features}  #
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
train_model_input["genres"] =  genres_list

model = DeepFEFM(linear_feature_columns, dnn_feature_columns, task='binary')

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['AUC', 'Precision', 'Recall'])
model.summary()

print('begain training')
model.fit(train_model_input, data[target].values,
						batch_size=256, epochs=20, verbose=2,
						validation_split=0.2
				)