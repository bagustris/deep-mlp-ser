#!/usr/bin/python
# Run inside MSP-IMRPROV dir
# Output: csv files with functionalsi of low-level descriptors (LLDs) + silence

import os
import fnmatch
import numpy as np
import librosa
import pandas as pd

# Folders with feature files
folder_lld_features = './egemaps_improv/'
#improv_path = '/media/bagustris/bagus/dataset/IEMOCAP_full_release/'

# Get all files, filenames are the same for all modalities
files = fnmatch.filter(os.listdir(folder_lld_features), '*.csv')
files.sort()
feat = []

# Generate files with functionals
for fn in files:
    print("Process ...", fn)
    data = pd.read_csv(folder_lld_features+fn, sep=';', usecols=range(2,25))
    #extract silence
    filename = 'session' + fn[18] + '/' + fn[11:15] + '/' \
               + fn[20] + '/' + fn[:-4] + '.wav'
    y, sr = librosa.load(filename) #, sr=16000)   
    rmse = librosa.feature.rms(y + 0.0001)[0]

    silence = 0
    for e in rmse:
        if e <= 0.3 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    silence_np = np.array(silence).reshape(1,)
    
    X_func = np.concatenate((np.array(data.mean()), np.array(data.std()), silence_np))
    feat.append(X_func)

feat = np.array(feat)
np.save('feat_hfs_msp3.npy', feat)
