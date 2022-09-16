#!/opt/conda/bin/python

import os
import glob
import time
import mne
import pickle
import numpy as np
#from mne.preprocessing import ICA

# From the data manual
BDF_PATH = "/net2/expData/affective_eeg/mahnob_dataset/Sessions"

# Taken from https://github.com/TNEL-UCSD/limbic-analysis-dynamics/blob/
#            master/scripts/iads_gen_features.py
FREQ_RANGES = {
    'delta': [2., 3.5],
    'theta': [4., 7.],
    'alpha': [9., 11.],
    'beta': [15., 30.],
    'gamma': [35., 55.],
    'high_gamma': [90., 110.]
}

FREQ_BANDS = ["theta", "alpha", "beta"]

EEG_CHS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 
           'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 
           'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 
           'CP2', 'P4', 'P8', 'PO4', 'O2']
STIM_CHS = ['Status']
OTHER_CHS = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 
             'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp']

def bdf_to_features(dataset_path=BDF_PATH):
    '''
    Preprocess data, find minimum data length, and store in file
    Warning: long run time.
    Must be run first
    Store in folder raw.
    
    arguments
    dataset_path: Optional. If not provided the default path 
                  /net2/expData/affective_eeg/mahnob_dataset/Sessions 
                  will be used.
    '''
    
    # Make bdf file path list
    print("Fetching bdf files.")
    dataset_path = "/net2/expData/affective_eeg/mahnob_dataset/Sessions"
    meta_data_path = "session.xml"
    all_session_nums = os.listdir(dataset_path) # List of all session names
    raw_names = [] # Absolute paths to all bdf files
    session_nums = [] # Sessions with bdf recordings
    # Get current working directory to change back later
    curr_dir = os.getcwd()
    # From data manual, bdf file may not exist if "the trials is missing due 
    # to technical difficulties" (pg 15).
    # Skip all sessions with no bdf recordings
    for session in all_session_nums:
        session_path = os.path.join(dataset_path, session)
        os.chdir(session_path)
        bdf_list = glob.glob("*.bdf")
        
        if (len(bdf_list) == 1):
            session_nums.append(session)
            name = os.path.join(dataset_path, session, bdf_list[0])
            raw_names.append(name)
        elif (len(bdf_list) > 1):
            raise ValueError("Cannot handle multiple bdfs in one session.")
            
    # Change back to notebook directory as a precaution
    os.chdir(curr_dir)
    print("Back to directory: ", os.getcwd())
    
    # Find the shortest recording for time cutoff
    print("Try to find the shortest recording.")
    print("Read in data for the first time. This may take a while.")
    start_time = time.time()
    loaded_raws = []
    min_record_len = 1000
    for session, raw_name in zip(session_nums, raw_names):
        raw_path = os.path.join(dataset_path, session, raw_name)
        raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel="Status", verbose=False)
        if (not len(events) == 2):
            raise ValueError("Found more than two events.")
            
        # Cannot find method of conversion. Use time index/sampling freq
        event_idxs = [events[x][0] for x in range(len(events))]
        sampling_freq = int(raw.info["sfreq"])
        start, end = [int(index / sampling_freq) for index in event_idxs]
        duration = end - start
        if duration < min_record_len:
            min_record_len = duration
        loaded_raws.append(raw)
    print("Took %ss to finish." % round(time.time() - start_time))
    print(format("Shortest stimuli: {}s".format(min_record_len)))
    
    # Preprocess raw
    print("Start preprocessing. This may take a while.")
    print("WARNING: no noise removal")
    start_time = time.time()
    # Check and make raw directory
    if not os.path.exists("raw"):
        os.makedirs("raw")
    # Cut, tranform to feature array, and store
    for i in range(len(loaded_raws)):
        raw = loaded_raws[i]
        
        # Find stimuli start and stop
        events = mne.find_events(raw, stim_channel="Status", verbose=False)
        start_idx = events[0][0]
        sampling_freq = int(raw.info["sfreq"])
        start = int(start_idx / sampling_freq)
        end = start + min_record_len
        
        # Cut data based on duration
        raw = raw.crop(tmin=start, tmax=end)
        # Drop channel, rereference, filter
        raw = raw.drop_channels(OTHER_CHS)
        raw = raw.drop_channels(STIM_CHS)
        raw, _ = mne.set_eeg_reference(raw, verbose=False) # Rereference by mean
        raw = raw.filter(l_freq=1, h_freq=49, verbose=False)
        
        # ICA
            #ica = ICA(n_components=64, random_state=413, verbose=False)
            #ica.fit(raw)
            #ica.exclude = [0, 1]
            #ica.apply(raw) # Project back
        
        # Store raw in pickle
        idx = "{:04}".format(i)
        file_name = "raw/session_" + idx + ".pkl"
        to_features(raw, file_name)
        
    print("Took %ss to finish." % round(time.time() - start_time))
    print("Done!")


def to_features(raw, file_name, verbose=False):
    
    """The preprocessing pipeline for one session eeg data. Frequency band 
    extraction and hilbert transform. Store feature array in pickle file.

    arguments
        raw         - pointer to the preprocessed and preloaded mne raw
        file_name   - relative location and pickle file name string
        verbose     - (not implemented) Whether to output progress and sample 
                      output
    """

    # Frequency bands extraction
    freq_bands = FREQ_BANDS
    raw_by_freq = []
    for i in range(len(freq_bands)):
        low = FREQ_RANGES[freq_bands[i]][0]
        high = FREQ_RANGES[freq_bands[i]][1]
        raw_in_freq = raw.copy().filter(l_freq=low, h_freq=high, verbose=False)
        raw_by_freq.append(raw_in_freq)

    # z-score & Hilbert transform
    # Based on code from limbic-analysis-dynamics repo
    for raw in raw_by_freq:
        data = raw.get_data()
        # mean across time per channel
        m = np.mean(data, axis=1)
        m = np.tile(m, (data.shape[1], 1)).T
        s = np.std(data, axis=1, ddof=1)
        s = np.tile(s, (data.shape[1], 1)).T
        z_score = (data - m) / s
        raw.apply_function(lambda _: z_score, channel_wise=False)
        raw.apply_hilbert(n_jobs=16, envelope=True, verbose=False)
        raw.apply_function(np.square, dtype=np.float64)

    # Get data and store to pickle
    data = []
    for raw in raw_by_freq:
        data.append(raw.get_data())
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp)
    
    
if __name__ == "__main__":
    bdf_to_features()