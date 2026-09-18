import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
