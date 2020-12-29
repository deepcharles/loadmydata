# loadmydata
Utility functions for loading data sets.

## UEA/UCR time series classification repository

The list of available data sets from UEA/UCR repository is available [here](http://www.timeseriesclassification.com/dataset.php).


## Usage

```python
from loadmydata.load_uea_ucr import load_uea_ucr_data

data = load_uea_ucr_data("ArrowHead")

print(data.description)

print(data.X_train.shape)
print(data.X_test.shape)
print(data.y_train.shape)
print(data.y_test.shape)
```