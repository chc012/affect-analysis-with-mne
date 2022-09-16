# Predicting self-reported emotion with EEG data

- `get_feature_response.py`: contains methods:
    1. bdf_to_features: extract features based on channel, frequency band, and time window;
    2. get_affect: get affective responses
- `mne_explore.ipynb`: exploratory analysis and preprocessing pipline of single session eeg data using Python-MNE.
- `model_explore.ipynb`: explore several feature extractions and predictions pipeline using LDA and SVC.
- `model_verify.ipynb`: check svc and lda models using irrelevant data
- `preprocess.py`: used to make .fif files for preprocessed data (long run time)

## Usage
- __`preprocess.py`__ must be run first to get all preprocessed data into `raw/` directory (about 600 MB upon completion).
- EEG data path of one session must be provided to `raw_path` in `mne_explore`.