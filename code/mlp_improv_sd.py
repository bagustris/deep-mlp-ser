# lstm_lstm: emotion recognition from speech= lstm, text=lstm
# created for ATSIT paper 2020
# coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)

import numpy as np
import pandas as pd
import random as sd

#import keras.backend as K
#from keras.models import Model, Sequential
#from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, \
#                         Bidirectional, Flatten, \
#                         Embedding, Dropout, Flatten, BatchNormalization, \
#                         RNN, concatenate, Activation
#from keras.callbacks import EarlyStopping
#from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neural_network import MLPRegressor
from calc_scores import calc_scores

rn.seed(123)
np.random.seed(99)

# loading gemaps feature file and label
feat = np.load('/home/s1820002/msp-improv/data/feat_hfs_msp3.npy')

list_path = '/home/s1820002/msp-improv/helper/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_sorted = list_file.sort_values(by=['wavfile'])
vad_list = [list_sorted['v'], list_sorted['a'], list_sorted['d']]
vad = np.array(vad_list).T

# standardization
scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
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
#def ccc(gold, pred):
#    gold       = K.squeeze(gold, axis=-1)
#    pred       = K.squeeze(pred, axis=-1)
#    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
#    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
#    covariance = (gold-gold_mean)*(pred-pred_mean)
#    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
#    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
#    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
#    return ccc


#def ccc_loss(gold, pred):  
#    # input (num_batches, seq_len, 1)
#    ccc_loss   = K.constant(1.) - ccc(gold, pred)
#    return ccc_loss

X_train = feat[:6570]
X_test = feat[6570:]
y_train = vad[:6570]
y_test = vad[6570:]

# batch_size=min(200, n_samples)
# layers (256, 128, 64, 32, 16)
nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=180, shuffle=True,
    random_state=9, verbose=0, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn = nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)

ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], y_test[:, i])
    ccc.append(ccc_)
    #print("# ", ccc)

print(ccc)
    
#Results:
# 0.4874353476028858
# 0.6822788332623598
# 0.5516393803700689

