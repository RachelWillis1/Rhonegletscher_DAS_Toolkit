# Rhonegletscher DAS Toolkit

This repository contains the Python codebase used in support of the manuscript titled:

**"Creating a Comprehensive Cryoseismic Catalog at Rhonegletscher: A Scalable Approach Using Distributed Acoustic Sensing and Machine Learning"**

## Overview

This code repository supports the feature extraction, ML, visualization, & initial analysis of cryoseismic events with Distributed Acoustic Sensing (DAS) data collected from Rhonegletscher, Switzerland. The core workflow involves:

- Preprocessing and covariance-based feature extraction from DAS data
- Comparison of supervised and unsupervised ML for event classification
- Visualization of cluster distributions
- Performance comparison of STA/LTA and ML-based detectors
- Initial dispersion analysis for cryoseismic events

## Directory Structure

```
Rhonegletscher_DAS_ML_Toolkit/
├── AWS_Rhone_Glacier_Steps         # Steps to run toolkit on AWS
├── data/                           # Cable layout info, sample features, training classification,
                                      random forest predictions, STA/LTA event predictions
├── results/                        # Full catalog of random forest predictions
├── mount_vols_create_conda_env.sh  # Bash script for AWS setup
├── random_forest_model.pkl         # Trained random forest model
├── README.md                       # This file
├── runx.sh                         # Main execution script (Adjust time range)
├── S3filelist_selectiontxt.py      # Create file list on AWS
├── S3foldefileliststxt.py          # Edit AWS S3 folder & file list
├── scripts-main/                   # Main Python scripts for ML, feature extraction, plotting
├── time_ranges_for_full_aws_run    # Time ranges used for AWS runs
```

## Key Scripts

| Script Name                                  | Description                              |
|----------------------------------------------|------------------------------------------|
| `feature_extraction_and_clustering_wDASK.py` | Performs feature extraction & clustering |
| `hyperparameter_gridsearch_ml.py`            | Grid search unsupervised & supervised ML |
| `par_file.py`                                | Parameters for preprocessing & STA/LTA   |
| `plotting_functions.py`                      | Visualizes for strain rate & covariance  |
| `preprocessing_functions.py`                 | Preprocessing functions for DAS data     |
| `readers.py`                                 | Functions to read different DAS data     |
| `stalta_event_detection.py`                  | STA/LTA function for event detection     |


## Dependencies

Ensure you have the following installed:
- Python 3.9
- Required packages:
  ```bash
  conda install numpy==1.26.4
  conda install matplotlib==3.9.2
  conda install pandas==2.2.3
  conda install scipy==1.12.0
  conda install h5py==3.12.1
  conda install scikit-learn==1.1.2
  pip install obspy==1.4.1
  pip install pyasdf==0.8.1
  pip install dask
  pip install "dask[distributed]" --upgrade
  pip install joblib==1.2.0
  ```


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/RachelWillis1/Rhonegletscher_DAS_Toolkit.git
   cd Rhonegletscher_DAS_Toolkit
   ```

2. Prepare repository:
   - If running on AWS follow steps and mount volumes.
   - Adjust file paths in par_file.

3. Run an analysis or visualization script:
   ```bash
   python scripts-main/feature_extraction_and_clustering_wDASK.py
   ```

4. View results:
   - Features and predictions will be saved to the output file.
   - Figures will be saved based to individual path in scripts.

## Citation

If you use this notebook or the associated methodology in your work, please cite any of our related publication:  

- *Willis, R. M., et al. (in review). Creating a comprehensive cryoseismic catalog at Rhonegletscher: A scalable approach using distributed acoustic sensing and machine learning. JGR: Machine Learning and Computation.*
- *Willis, R. M. (2025). From Injection to Ice: Advancing Climate Solutions Through Seismology, Machine Learning, and Open Science (Doctoral dissertation). Colorado School of Mines.*
