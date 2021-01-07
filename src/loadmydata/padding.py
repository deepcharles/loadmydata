import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskedArray


def pad_at_the_end(signal: np.ndarray, pad_width: int) -> MaskedArray:

    assert pad_width >= 0, f"pad_width (={pad_width}) must be positive."

    if signal.ndim == 1:
        (n_samples,) = signal.shape
        n_dims = 1
    else:
        n_samples, n_dims = signal.shape

    return ma.masked_array(
        data=np.pad(
            signal.reshape(n_samples, n_dims).astype(np.float),
            pad_width=((0, pad_width), (0, 0)),
            mode="constant",
            constant_values=(np.nan,),
        ),
        mask=np.tile([False] * n_samples + [True] * pad_width, (n_dims, 1)).T,
    )


def get_signal_shape(signal_padded: MaskedArray) -> (int, int):
    err_msg = "Wrong dimensions: {signal_padded.shape}. Expected: (n_samples, n_dims)."
    assert signal_padded.ndim == 2, err_msg

    padded_size, n_dims = signal_padded.shape
    n_samples = padded_size - signal_padded.mask[:, 0].sum()
    return (n_samples, n_dims)
