from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array as JXArray
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import DTypeLike


class TimeSeriesDataset:

    def __init__(self, input_dim: np.ndarray, output_dim: np.ndarray, data: list[np.ndarray],
                       target_feature: int = 0) -> None:
        self._idim = input_dim
        self._odim = output_dim
        self._data = data

        self._n_features = -1
        for _seq in data:
            if self._n_features == -1:
                self._n_features = _seq.shape[1]
            elif _seq.shape[1] != self._n_features:
                raise ValueError("Data series must have same amount of features")
        if target_feature >= self._n_features:
            raise ValueError("Target feature exceeds number of features")
        self._target = target_feature

        self._calculate_xy()

    def _calculate_xy(self):
        _x = []
        _y = []
        for _seq in self._data:
            # Rolling window
            _x.append(sliding_window_view(_seq[:-self._odim], self._idim, axis=0).reshape((-1,self._idim*self._n_features)))
            _y.append(sliding_window_view(_seq[self._idim:,self._target], self._odim, axis=0))
        self._x = np.concatenate(_x, axis=0)
        self._y = np.concatenate(_y, axis=0)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def target_feature(self) -> int:
        return self._target

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()


def _unpack(dataset: Any, target: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract the raw input and target arrays of a scikit-learn dataset.

    Args:
        dataset (Any): Either a scikit-learn ``Bunch`` (anything with ``data``
            and ``target`` members, as returned by ``sklearn.datasets.load_*``),
            an ``(X, y)`` pair (as returned with ``return_X_y=True``), or the
            input array itself when ``target`` is given.
        target (Any, optional): The target array, when ``dataset`` holds only
            the inputs. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: The input array, of shape
        ``(n_samples, n_features)``, and the raw target array.

    Raises:
        ValueError: If the targets can not be found, or if inputs and targets
            have different numbers of samples.
    """
    if target is not None:
        x, y = dataset, target
    elif hasattr(dataset, "data") and hasattr(dataset, "target"):
        x, y = dataset.data, dataset.target
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        x, y = dataset
    else:
        raise ValueError(
            "Dataset must be a scikit-learn Bunch, an (X, y) pair, or an input "
            "array paired with an explicit target argument"
        )

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(
            f"Inputs and targets have different lengths: {len(x)} and {len(y)}"
        )

    return x, y


def sklearn_classification_data(
    dataset: Any,
    target: Any | None = None,
    one_hot: bool = True,
    n_classes: int | None = None,
    dtype: DTypeLike = jnp.float32,
) -> tuple[JXArray, JXArray]:
    """Convert a scikit-learn classification dataset into JAX arrays.

    Class labels of any type (integers, strings) are mapped onto the indices
    0 to ``n_classes-1``, in ascending order of the labels themselves. Only the
    classes present in the data are counted, so pass ``n_classes`` when
    converting a subset that may not hold them all.

    Args:
        dataset (Any): A scikit-learn ``Bunch``, an ``(X, y)`` pair, or the
            input array itself when ``target`` is given.
        target (Any, optional): The class labels, when ``dataset`` holds only
            the inputs. Defaults to None.
        one_hot (bool, optional): Whether to one-hot encode the labels, giving
            targets of shape ``(n_samples, n_classes)``. If False, the class
            indices are returned as an integer array of shape ``(n_samples,)``,
            as expected by a loss built with ``expect_labels=True``. Defaults
            to True.
        n_classes (int, optional): Total number of classes. Inferred from the
            labels if not given. Defaults to None.
        dtype (DTypeLike, optional): Data type of the returned arrays. Applies
            to the targets only if they are one-hot encoded. Defaults to
            ``jnp.float32``.

    Returns:
        tuple[JXArray, JXArray]: The inputs, of shape
        ``(n_samples, n_features)``, and the targets.

    Raises:
        ValueError: If ``n_classes`` is smaller than the number of classes
            found in the labels.
    """
    x, y = _unpack(dataset, target)

    # Map arbitrary labels onto 0...n_found-1
    classes, labels = np.unique(y, return_inverse=True)
    n_found = len(classes)
    if n_classes is None:
        n_classes = n_found
    elif n_classes < n_found:
        raise ValueError(
            f"Labels hold {n_found} classes, more than the {n_classes} declared"
        )

    x_jx = jnp.array(x, dtype=dtype)
    if not one_hot:
        return x_jx, jnp.array(labels, dtype=jnp.int32)

    y_jx = jnp.zeros((len(labels), n_classes), dtype=dtype)
    y_jx = y_jx.at[jnp.arange(len(labels)), jnp.array(labels)].set(1)

    return x_jx, y_jx


def sklearn_regression_data(
    dataset: Any,
    target: Any | None = None,
    dtype: DTypeLike = jnp.float32,
) -> tuple[JXArray, JXArray]:
    """Convert a scikit-learn regression dataset into JAX arrays.

    Targets are always returned two-dimensional, so that single and multiple
    output problems have the same shape convention.

    Args:
        dataset (Any): A scikit-learn ``Bunch``, an ``(X, y)`` pair, or the
            input array itself when ``target`` is given.
        target (Any, optional): The target values, when ``dataset`` holds only
            the inputs. Defaults to None.
        dtype (DTypeLike, optional): Data type of the returned arrays. Defaults
            to ``jnp.float32``.

    Returns:
        tuple[JXArray, JXArray]: The inputs, of shape
        ``(n_samples, n_features)``, and the targets, of shape
        ``(n_samples, n_targets)``.
    """
    x, y = _unpack(dataset, target)

    if y.ndim == 1:
        y = y[:, None]

    return jnp.array(x, dtype=dtype), jnp.array(y, dtype=dtype)
