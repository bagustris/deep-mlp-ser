# extract_34.py: extract 34 acoustic features from IEMOCAP dataset

import numpy as np
from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import ShortTermFeatures
#from keras.preprocessing import sequence
import glob
import os
import ntpath
import librosa

data_path = '/media/bagus/data01/dataset/MSP-IMPROV/'

# this list of wav files is consistent with labels
# checked with == operator (data_id == files_id)
files = glob.glob(os.path.join(data_path + 'session?/*/?/', '*.wav'))
files.sort(key=lambda x: x[-30:])  

hsf_train = []
hsf_test = []

for f in files:
    if int(ntpath.basename(f)[18]) in range(1, 6):
        print("Processing...", f)
        [Fs, x] = audioBasicIO.read_audio_file(f)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025*Fs, 
                 0.010*Fs, deltas=False)
        mean_train = np.mean(F, axis=1)
        std_train = np.std(F, axis=1)
        
        # extract silence
        # duration, 1 s = 44100 samples, 200 ms = 8820, 500 ms = 22050
        rmse = librosa.feature.rms(x + 0.0001, frame_length=44100)[0]
        silence = 0
        for e in rmse:
            if e <= 0.3 * np.mean(rmse):
                silence += 1
        silence /= float(len(rmse))
        silence_np = np.array(silence).reshape(1,)
        feat_hsf_train = np.hstack([mean_train, std_train, silence_np])
        hsf_train.append(feat_hsf_train)
        
    elif int(ntpath.basename(f)[18])==6:
        print("Processing...", f)
        [Fs, x] = audioBasicIO.read_audio_file(f)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025*Fs, 
                 0.010*Fs, deltas=False)
        mean_test = np.mean(F, axis=1)
        std_test = np.std(F, axis=1)
        
        # extract silence
        # duration, 1 s = 44100 samples, 200 ms = 8820, 500 ms = 22050
        rmse = librosa.feature.rms(x + 0.0001, frame_length=44100)[0]
        silence = 0
        for e in rmse:
            if e <= 0.3 * np.mean(rmse):
                silence += 1
        silence /= float(len(rmse))
        silence_np = np.array(silence).reshape(1,)
        feat_hsf_test = np.hstack([mean_train, std_test, silence_np])
        hsf_test.append(feat_hsf_test)

# save feature
np.save('data/feat_34_msp_train_1000.npy', np.array(hsf_train))
np.save('data/feat_34_msp_test_1000.npy', np.array(hsf_test))
