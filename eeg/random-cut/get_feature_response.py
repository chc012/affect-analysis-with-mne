import random
import numpy as np
import xml.etree.cElementTree as et
#from mne.preprocessing import ICA

def random_cut(data, duration=35, cut_percent=0, cut_window=0.5):

    """Resample and return data with randomized time window cut.

    arguments
        data        - features list
        duration    - stimuli length and recording used in seconds
        cut_percent - Percentage of randomly omitted data. Will approximate 
                      to the closest number of cuts with the cut_window size
        cut_window  - Window size of each cut in seconds. Default is 0.5.

    returns
        features         - 2D array with freq bands, average channel power, 
                           and time dims
        features_red_dim - 1D array from squeezing features
    """
    
    # enforce conversion (uncertain about this)
    data = np.array(data)
    num_band = len(data)
    num_ch = len(data[0])
    num_idx = len(data[0][0])
    
    
    # Resampling and randomly cutting out some windows
    num_sample = int(duration / cut_window)
    idx_per_sample = int(num_idx / num_sample)
    sample_seps = [x * idx_per_sample for x in range(num_sample)]
    # Random window selection
    num_left = round(num_sample * (1 - cut_percent)) # Windows NOT cut away
    sample_seps = random.sample(sample_seps, num_left)
    sample_seps.sort()
    
    # Magical numpy vectorized calculation to speed up
    # Make feature array, but only store channel power average this time
    data_post_cut = np.empty((num_band, num_ch, num_left))
    for i in range(len(sample_seps)):
        start = sample_seps[i]
        end = sample_seps[i] + idx_per_sample
        data_post_cut[:,:,i] = np.apply_along_axis(np.mean, 2, 
                                                   data[:,:,start:end])
    features = np.apply_along_axis(np.mean, 2, data_post_cut)
    
    #features_list = []
    #for band in data:
    #    resampled_band = []
    #    for ch in band:
    #        # To average over channel power
    #        resampled_ch = []
    #        for i in range(len(sample_seps)):
    #            start = sample_seps[i]
    #            end = sample_seps[i] + idx_per_sample
    #            sample = np.mean(ch[start:end])
    #            resampled_ch.append(sample)
    #        resampled_band.append(np.mean(resampled_ch))
    #    features_list.append(resampled_band)
    # Make feature arrays
    #features = np.array(features_list)
    
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
    
    # xml magics
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