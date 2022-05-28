'''
Author: your name
Date: 2022-03-21 10:59:33
LastEditTime: 2022-03-31 16:03:12
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/FM/train.py
'''
"""
Created on August 25, 2020

train FM model

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import DeepFM
from criteo import create_criteo_dataset
from utils import getFeature

import os

# tf.config.experimental.list_physical_devices(device_type='CPU')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.config.experimental.list_physical_devices(device_type='CPU')


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    data_file = '/home/nesa320/huizhong/fm/dataset/criteo/train.txt'

    check_path = '/home/nesa320/huizhong/deepfm/save/deepfm-v1000-622.ckpt'

    read_part = True
    sample_num = 1000 * 10000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 1000
    epochs = 20

    
    # ========================== Create dataset =======================
    feature_columns, train, test, tune = create_criteo_dataset(file=data_file,
                                           read_part=read_part,
                                           sample_num=sample_num,
                                           test_size=test_size)
    f = open('/home/nesa320/huizhong/deepfm/feat.txt' , 'a')
    print(feature_columns,file=f)
    train_x, train_y = train
    test_x, test_y = test
    
    tune.to_csv('/home/nesa320/huizhong/deepfm/dataset/test/tune.csv')
    np.savetxt('/home/nesa320/huizhong/deepfm/dataset/test/test_x.npy',test_x,fmt='%d')
    np.savetxt('/home/nesa320/huizhong/deepfm/dataset/test/test_y.npy',test_y,fmt='%d')
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
        model.summary()
        # ============================Compile============================
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, period=5)
    # ==============================Fit==============================
    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        callbacks=[checkpoint],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_x, test_y, batch_size=batch_size)[1])