import numpy as np

import tensorflow as tf
import xarray as xr

from typing import Union, List


def compute_drag_coefficient(U: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    constant_C_d = 1.61974e-2
    # C_d = np.where(U >= 50.33, constant_C_d, (-0.16 * U ** 2 + 0.967 * U + 8.058) * 1e-4)
    C_d = tf.where(
        U >= 50.33, constant_C_d, (-0.16 * tf.math.square(U) + 0.967 * U + 8.058) * 1e-4
    )
    return C_d


def compute_wind_stress(
    U_x: Union[tf.Tensor, np.ndarray],
    U_y: Union[tf.Tensor, np.ndarray],
    rho: float = 1.28,
) -> tf.Tensor:
    U = tf.math.sqrt(tf.math.square(U_x) + tf.math.square(U_y))
    C_d = compute_drag_coefficient(U)
    tau = rho * C_d * tf.math.square(U)
    return tau


def normalize_and_clean(
    tensor: tf.Tensor, var_name: str, min_max_ds: xr.Dataset
) -> tf.Tensor:
    """Normalize tensor and replace NaNs with -9."""
    min_val = min_max_ds[var_name].values[0]
    max_val = min_max_ds[var_name].values[1]
    tensor = (tensor - min_val) / (max_val - min_val)
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor) - 9, tensor)
    return tensor


def reshape_channels(tensor: tf.Tensor, indices: List[int]) -> tf.Tensor:
    """Slice off first channel and reshape to (481, 421, 2)."""
    # index 0 is -12H, 1 is -6H, 2 is now
    tensor = tf.gather(tensor, indices=indices, axis=-1)
    return tf.reshape(tensor, (481, 421, len(indices)))
