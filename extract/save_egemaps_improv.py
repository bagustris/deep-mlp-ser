#!/usr/bin/env python3
# bagus@ep.its.ac.id, 2019-04-16

import numpy as np
import os
import time
import ntpath
import pickle

feature_type = 'egemaps'
exe_opensmile = '~/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'  
path_config   = '~/opensmile-2.3.0/config/'                                      
improv_path = '/media/bagus/data01/dataset/MSP-IMPROV/'

if feature_type=='mfcc':
    folder_output = './mfcc_improv/'  # output folder
    conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
    outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
elif feature_type=='egemaps':
    folder_output = './egemaps_improv/'  # output folder
    conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
    outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc

else:
    print('Error: Feature type ' + feature_type + ' unknown!')

if not os.path.exists(folder_output):
    os.mkdir(folder_output)
    
with open(improv_path + 'Evaluation.txt') as f:
    for line in f:
        if line[:3] == 'UTD':
            label = line.split(';')
            instname = 'MSP-'+ label[0][4:-4]
            filename = improv_path + 'session' + instname[18] + '/' + instname[11:15] \
                         + '/' + instname[20] + '/' + instname + '.wav'
            outfilename = folder_output + instname + '.csv'
            opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + filename + ' ' + outputoption + ' ' + outfilename + ' -instname ' + instname + ' -output ?'  # (disabling htk output
            os.system(opensmile_call)
            time.sleep(0.01)

os.remove('smile.log')
