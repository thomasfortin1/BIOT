import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

# we need these channels
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F7-REF']],  # 0
# (signals[signal_names['EEG F7-REF']] - signals[signal_names['EEG T3-REF']]),  # 1
# (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG T5-REF']]),  # 2
# (signals[signal_names['EEG T5-REF']] - signals[signal_names['EEG O1-REF']]),  # 3
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F8-REF']]),  # 4
# (signals[signal_names['EEG F8-REF']] - signals[signal_names['EEG T4-REF']]),  # 5
# (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG T6-REF']]),  # 6
# (signals[signal_names['EEG T6-REF']] - signals[signal_names['EEG O2-REF']]),  # 7
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F3-REF']]),  # 14
# (signals[signal_names['EEG F3-REF']] - signals[signal_names['EEG C3-REF']]),  # 15
# (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG P3-REF']]),  # 16
# (signals[signal_names['EEG P3-REF']] - signals[signal_names['EEG O1-REF']]),  # 17
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F4-REF']]),  # 18
# (signals[signal_names['EEG F4-REF']] - signals[signal_names['EEG C4-REF']]),  # 19
# (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG P4-REF']]),  # 20
# (signals[signal_names['EEG P4-REF']] - signals[signal_names['EEG O2-REF']]))) # 21
channels_to_keep = [
    'EEG FP1-REF',
     'EEG FP2-REF',
     'EEG F4-REF',
     'EEG C3-REF',
     'EEG C4-REF',
     'EEG P3-REF',
     'EEG P4-REF',
     'EEG O1-REF',
     'EEG O2-REF',
     'EEG F7-REF',
     'EEG F8-REF',
     'EEG T3-REF',
     'EEG T4-REF',
     'EEG T5-REF',
     'EEG T6-REF',
     'EEG F3-REF',
     'EEG CZ-REF',
     'EEG FZ-REF',
     'EEG PZ-REF',
     'EEG T2-REF',
     'EEG T1-REF',
     'EEG EKG1-REF',
     'EEG A1-REF',
     'EEG A2-REF',
]


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.resample(256)
            try:
                channeled_data = raw.get_data(channels_to_keep)
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2560):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2560 : (i + 1) * 2560], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    # root to abnormal dataset
    root = "/media/data_ssd/data/tuab/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v3.0.1/edf"
    channel_std = "01_tcp_ar"
    new_root = "proccessed_mine"

    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, new_root)):
        os.makedirs(os.path.join(root, new_root))

    if not os.path.exists(os.path.join(root, new_root, "train")):
        os.makedirs(os.path.join(root, new_root, "train"))
    train_dump_folder = os.path.join(root, new_root, "train")

    if not os.path.exists(os.path.join(root, new_root, "val")):
        os.makedirs(os.path.join(root, new_root, "val"))
    val_dump_folder = os.path.join(root, new_root, "val")

    if not os.path.exists(os.path.join(root, new_root, "test")):
        os.makedirs(os.path.join(root, new_root, "test"))
    test_dump_folder = os.path.join(root, new_root, "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    for val_sub in val_a_sub:
        parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, test_dump_folder, 0])

    # split and dump in parallel
    with Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)
