# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2012-12-25: modified fot ser_iemocap_loso_hfs.py
#             feature is either std+mean or std+mean+silence (uncomment line 44)

import numpy as np
import pickle
import pandas as pd

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, TimeDistributed, \
                         Bidirectional, Flatten, Convolution1D, \
                         Embedding, Dropout, Flatten, BatchNormalization, \
                         RNN, concatenate, Activation

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
feat_iemocap = np.load('/home/s1820002/spro2020/data/feat_ws_3.npy')
vad_iemocap = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/y_egemaps.npy')

feat_improv_train = np.load('/home/s1820002/deepMLP/data/feat_hfs_gemaps_msp_train.npy')
feat_improv_test = np.load('/home/s1820002/deepMLP/data/feat_hfs_gemaps_msp_test.npy')

feat_improv = np.vstack([feat_improv_train, feat_improv_test])

list_path = '/home/s1820002/msp-improv/helper/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_sorted = list_file.sort_values(by=['wavfile'])
vad_list = [list_sorted['v'], list_sorted['a'], list_sorted['d']]
vad_improv = np.array(vad_list).T

# for LSTM input shape (batch, steps, features/channel)
feat = np.vstack([feat_iemocap, feat_improv])
vad = np.vstack([vad_iemocap, vad_improv])

# for LSTM input shape (batch, steps, features/channel)
feat = np.vstack([feat_iemocap, feat_improv])
vad = np.vstack([vad_iemocap, vad_improv])

# for CNN input shape (batch, channel, steps)
feat = feat.reshape(feat.shape[0], feat.shape[1], 1)

# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)

# standardization
scaled_feature = False

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss


# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = Convolution1D(256, 3, activation='relu')(net_speech)
    net_speech = Convolution1D(128, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(64, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(32, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(64, 12, activation='relu')(net_speech)
    model_speech = Flatten()(net_speech)
    #model_speech = Dropout(0.1, seed=None)(net_speech)

    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(model_speech) for name in target_names]
    #model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model(input_speech, model_combined) 
    #model.compile(loss=ccc_loss, optimizer='rmsprop', metrics=[ccc])
    model.compile(loss=ccc_loss, 
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='adam', metrics=[ccc])
    return model

#def main(alpha, beta, gamma):
model = api_model(0.1, 0.5, 0.4)
model.summary()

idx_train = np.hstack([np.arange(0, 7869), np.arange(10039, len(feat_improv_train))])
idx_test = np.hstack([np.arange(7869,10039), np.arange(10039 + 
           len(feat_improv_train), 18387)])

# 8000 first data of session 5 (for LOSO)
# for this cnn, using batch_size 32 is better than 200 (as MLP)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit(feat[idx_train], vad[idx_train].T.tolist(), batch_size=200, #best:8
                  validation_split=0.2, epochs=180, verbose=1, shuffle=True,
                  callbacks=[earlystop])
metrik = model.evaluate(feat[idx_test], vad[idx_test].T.tolist())
print(metrik)
print("CCC ave= ", np.mean(metrik[-3:]))
