#!/usr/bin/python
# Put the scripts into a subfolder of the AVEC2018_CES package, e.g., AVEC2018_CES/scripts_CES/
# Output: csv files with functionals of low-level descriptors (LLDs)

import os
import fnmatch
import numpy as np
import librosa
import pandas as pd

# Folders with feature files
folder_lld_features = './egemaps_improv/'
improv_path = '/media/bagus/data01/dataset/MSP-IMPROV/'

# Get all files
files = fnmatch.filter(os.listdir(folder_lld_features), '*.csv')  # filenames are the same for all modalities
files.sort()
feat = []

# Generate files with functionals
for fn in files:
    print("Processing ...", fn)
    # extract 23 features from GeMAPS
    data = pd.read_csv(folder_lld_features+fn, sep=';', usecols=range(2, 25))
    
    # extract silence
    filename = ('session' + fn[18] + '/' + fn[11:15] + '/'
               + fn[20] + '/' + fn[:-4] + '.wav')
    y, sr = librosa.load(filename, sr=16000)  
    # extract rms using different duration: 200 ms (3200), 500 ms (8000), 
    # and 1 s (16000 samples), 128 ms (2048) 
    rmse = librosa.feature.rms(y + 0.0001, frame_length=2048)[0]

    silence = 0
    for e in rmse:
        if e <= 0.3 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    silence_np = np.array(silence).reshape(1,)
    
    X_func = np.concatenate((np.array(data.mean()), np.array(data.std()), silence_np))
    feat.append(X_func)

feat = np.array(feat)
np.save('./data/msp_feat_ws_128.npy', feat)
