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