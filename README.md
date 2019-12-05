# Spectral Super-Resolution with Optimized Bands
This repository contains MATLAB code for vegetation paramter estimation from hyperspectral data using single/multi-task Gaussian process with covariance functions based on well-established spectral comparison metrics.

### Usage
```bash
# cd code 
```
#### Download the dataset
```bash
# python download_dataset.py
```
This script is designed to download and install dataset from the outdoors natural dataset (http://icvl.cs.bgu.ac.il/hyperspectral/) used in our paper. 

#### Training 
```bash
# python run_BO.py
```

#### Testing 
```bash
# python test_cnn.py
```
### Computing performance metrics
```bash
# python compute_metrics.py
```

## Citation
Please cite the following article if you use the code in this repository: 

U. B. Gewali, S. T. Monteiro and E. Saber, "Spectral super-resolution with optimized bands," Remote Sensing, 2019. [[article](https://doi.org/10.3390/rs11141648)][[bibtex](citation.bib)]

## Contact
Utsav Gewali (ubg9540@rit.edu)
