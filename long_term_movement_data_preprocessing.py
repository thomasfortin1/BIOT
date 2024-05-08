import wfdb
import numpy as np
import os

preprocessed_path = "/home/workplace/thomas/BIOT/unlabeled_long_term_movement"

for n in range(45):
    subject = f'FL{n:03}'
    try:
        signals2, fields2 = wfdb.rdsamp(subject, channels=[0, 1, 2], pn_dir='ltmm/')
        print(f'got it {subject}')
    except:
        print(f'error for {subject}')
        continue
    os.mkdir(os.path.join(preprocessed_path, subject))
    for i in range(0, len(signals2) - 1000, 1000):
        arr = signals2[i:i+1000,:]
        np.save(os.path.join(preprocessed_path, subject, str(i)), arr)
    