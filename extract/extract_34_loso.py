# extract_34.py: extract 34 acoustic features from IEMOCAP dataset

import numpy as np
from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence
import glob
import os
import ntpath


data_path = '/media/bagustris/bagus/dataset/MSP-IMPROV/'

# this list of wav files is consistent with labels
# checked with == operator (data_id == files_id)
#files = [os.path.basename(x) for x in glob.glob(os.path.join(data_path + './session?/*/?/', '*.wav'))]
files = glob.glob(os.path.join(data_path + './session?/*/?/', '*.wav'))
files.sort(key=lambda x: x[-30:])  

# feat_train = []
# feat_test = []
hfs_train = []
hfs_test = []

for f in files:
    if int(ntpath.basename(f)[18]) in range(1, 6):
        print("Process..., ", f)
        [Fs, x] = audioBasicIO.readAudioFile(f)
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 
                     0.025*Fs, 0.010*Fs)
        mean_train = np.mean(F, axis=1)
        std_train = np.std(F, axis=1)
        feat_hfs_train = np.hstack([mean_train, std_train])
        hfs_train.append(feat_hfs_train)
        feat_train.append(F.transpose())
        
    elif int(ntpath.basename(f)[18])==6:
        print("Process..., ", f)
        [Fs, x] = audioBasicIO.readAudioFile(f)
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.025*Fs, 
                 0.010*Fs)
        mean_test = np.mean(F, axis=1)
        std_test = np.std(F, axis=1)
        feat_hfs_test = np.hstack([mean_test, std_test])
        hfs_test.append(feat_hfs_test)
        feat_test.append(F.transpose())

# pad sequence for LLD feature
feat_train = sequence.pad_sequences(feat_train, dtype='float32', maxlen=3189)
feat_test = sequence.pad_sequences(feat_test, dtype='float32', maxlen=3189)

# pad train and test
#feat_train_pad[feat_train_pad==0] =np.nan
#feat_test_pad[feat_test_pad==0] =np.nan

# compute mean and std train
#for i in feat_train:
#    print("process...", i)
#    mean_train = np.mean(i, axis=0)
#    std_train = np.std(i, axis=0)
#    feat_hfs_train = np.hstack([mean_train, std_train])
#    hfs_train.append(feat_hfs_train)

np.save('data/feat_hfs_msp_train.npy', np.array(hfs_train))
np.save('data/feat_paa_msp_train.npy', np.array(feat_train))

## test
#for i in feat_test:
#    mean_test = np.mean(i, axis=0)
#    std_test = np.std(i, axis=0)
#    feat_hfs_test = np.hstack([mean_test, std_test])
#    hfs_test.append(feat_hfs_test)

np.save('data/feat_hfs_msp_test.npy', np.array(hfs_test))
np.save('data/feat_paa_msp_test.npy', np.array(feat_test))
