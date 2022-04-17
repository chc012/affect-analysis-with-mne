import mne
import numpy as np
import xml.etree.cElementTree as et
from mne.preprocessing import ICA

# From the data manual
EEG_CHS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 
           'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 
           'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 
           'CP2', 'P4', 'P8', 'PO4', 'O2']
STIM_CHS = ['Status']
OTHER_CHS = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 
             'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp']

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

def bdf_to_features(raw, duration, freq_bands, chs, window):

    """The preprocessing pipeline for one session eeg data. Please refer to the 
    notebook `eeg_mne_explore.ipynb` for stepwise details.

    arguments
        raw        - pointer to the preprocessed and preloaded mne raw
        duration   - stimuli length and recording used in second
        freq_bands - list of frequency bands in FREQ_RANGES extracted
        chs        - list of channels in EEG_CHS extracted
        window     - length of time sample extracted in seconds

    returns
        features         - 3D array with freq bands, channels, and time dims
        features_red_dim - 1D array from squeezing features array
    """

    # Channel extraction
    raw = raw.copy().pick_channels(chs)

    # Frequency bands extraction
    raw_by_freq = []
    for i in range(len(freq_bands)):
        low = FREQ_RANGES[freq_bands[i]][0]
        high = FREQ_RANGES[freq_bands[i]][1]
        raw_in_freq = raw.copy().filter(l_freq=low, h_freq=high)
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
        raw.apply_hilbert(n_jobs=16, envelope=True)
        raw.apply_function(np.square, dtype=np.float64)

    # Get data
    data = []
    for raw in raw_by_freq:
        data.append(raw.get_data())

    # Resampling
    num_idx = len(data[0][0])
    num_sample = int(duration / window)
    idx_per_sample = int(num_idx / num_sample)
    sample_seps = [x * idx_per_sample for x in range(num_sample + 1)]

    # Make feature array
    features_list = []
    for band in data:

        resampled_band = []
        for ch in band:

            resampled_ch = []
            for i in range(len(sample_seps) - 1):

                start = sample_seps[i]
                end = sample_seps[i+1]
                sample = np.mean(ch[start:end])
                resampled_ch.append(sample)

            resampled_band.append(resampled_ch)

        features_list.append(resampled_band)

    # Make feature arrays
    features = np.array(features_list)
    features_red_dim = features.flatten()
    
    return features, features_red_dim



def get_affect(xml_path, cutoff=5):

    """Get affective response into four categories:
           - low valence low arousal (0, 0)
           - low valence high arousal (0, 1)
           - high valence low arousal (1, 0)
           - high valence high arousal (1, 1)

    arguments
        xml_path - (absolute) path of the metadata file
        cutoff   - score above this is considered high

    returns
        result - int in 1-4, representing the four categories
    """

    tree = et.parse(xml_path)
    root = tree.getroot()
    va_results = [valence, arousal] = [int(root.attrib["feltVlnc"]),
                                       int(root.attrib["feltArsl"])]

    if (valence <= cutoff) and (arousal <= cutoff):
        result = (0, 0)
    elif (valence <= cutoff) and (arousal > cutoff):
        result = (0, 1)
    elif (valence > cutoff) and (arousal <= cutoff):
        result = (1, 0)
    else:
        result = (1, 1)

    return result