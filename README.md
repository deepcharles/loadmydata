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

Alternatively, you can use `conda`:

```
conda config --add channels conda-forge
conda install loadmydata
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
    # signal is a masked array of shape (T, d).
    # The true number of samples of the signal (without extra padding)
    # can be accessed with `get_signal_shape`.
    n_samples, n_dims = get_signal_shape(signal)
    # To get the signal without the extra padding, do
    signal_without_padding = signal[:n_samples]
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

## NYC taxi data set

This data set contains the number of New York taxi passengers aggregated in 30 minutes buckets for the period between July 2014 and January 2015. There are five anomalies occur during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow storm.

The raw data is from the NYC Taxi and Limousine Commission [1] and has been curated by [2].

[1]: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
[2]: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing.

### Usage

```
from loadmydata.load_nyc_taxi import load_nyc_taxi_dataset

X, y, description = load_nyc_taxi_dataset()

print(description)
```