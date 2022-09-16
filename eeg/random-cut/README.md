# Baseline emotion prediction with EEG data with randomized time window cut

- `get_feature_response.py`: contains methods:
    1. random_cut: extract features with random cut
    2. get_affect: get affective responses
- `random_cut_results.ipynb`: generate prediction result with LDA. Store results in `results/` (long run time, took 7 hours in last try)
- `preprocess.py`: used to make pickle files for preprocessed data, will generate about 3.5 GBs of data.

## Usage
- __`preprocess.py`__ must be run first to get all preprocessed data into `raw/` directory.