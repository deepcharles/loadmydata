# loadmydata
Utility functions for loading data sets (Python 3.7++)

The list of available data sets currently includes:

- the UEA/UCR repository.

This package relies on requests, tqdm, yarl (for the download), scipy and pandas (data container).

## UEA/UCR time series classification repository

The list of available data sets from UEA/UCR repository is available [here](http://www.timeseriesclassification.com/dataset.php).

## Install

Use `pip` to install.

```
pip install loadmydata
```

## Usage

```python
from loadmydata.load_uea_ucr import load_uea_ucr_data

dataset_name = "ArrowHead"  # "AbnormalHeartbeat", "ACSF1", etc. 
data = load_uea_ucr_data(dataset_name)

print(data.description)
print(data.X_train.shape)
print(data.X_test.shape)
print(data.y_train.shape)
print(data.y_test.shape)
```
