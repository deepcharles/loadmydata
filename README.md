# loadmydata

Utility functions for loading **time series** data sets (Python 3.7++).

The list of available data sets currently includes:

- the UEA/UCR repository.


## Install
This package relies on requests, tqdm, yarl (for the download), and numpy.

Use `pip` to install.

```
pip install loadmydata
```

## Data format

Consider a data set of *N* time series **y**<sup>(1)</sup>, **y**<sup>(2)</sup>,..., **y**<sup>(N)</sup>.
Each **y**<sup>(n)</sup> has *T*<sup>(n)</sup> samples and *d* dimensions.
Note that time series can have variable lengths, i.e. different *T*<sup>(n)</sup> but they share the same dimensionality *d*.

Such a data set is contained in a `numpy` array of shape (*N*, *T*, *d*) where *T*:=max<sub>n</sub> *T*<sup>(n)</sup>.
Time series with less than *T* samples are padded at the end with `numpy.nan`.
In addition, the extra padding is masked using [numpy's MaskedArray](https://numpy.org/doc/stable/reference/maskedarray.html).

```python
from loadmydata.padding import get_signal_shape

# Assume that X contains a time series data set of shape (N, T, d)
for signal in X:
    # signal is a masked array of shape (N, T).
    # The true number of samples of the signal (without extra padding)
    # can be accessed with `get_signal_shape`.
    n_samples, n_dims = get_signal_shape(signal)
    # To get the signal without the extra padding, do
    signal_without_padding = signal[~signal.mask]
    # do something with signal_without_padding
    ...
```

## UEA/UCR time series classification repository

The UEA/UCR repository focuses on time series classification.
As a result, each signal is associated with a label to predict.

The list of available data sets from the UEA/UCR repository is available [here](http://www.timeseriesclassification.com/dataset.php).


### Usage example

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
