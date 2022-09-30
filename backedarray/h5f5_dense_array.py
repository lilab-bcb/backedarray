import numpy as np

from backedarray.dense_array import DenseArray
from backedarray.index import Index
from backedarray.utils import register_dataset


class Hdf5DenseArray(DenseArray):
    def __init__(self, dataset: "h5py.Dataset"):
        super().__init__(dataset)

    def __getitem__(self, idx: Index):
        d = self.dataset
        # h5py requires indices be provided in ascending order
        if isinstance(idx, tuple):
            ordered = list(idx)
            rev_order = [slice(None) for _ in range(len(idx))]
            order_changed = False
            for axis, axis_idx in enumerate(ordered.copy()):
                if isinstance(axis_idx, (np.integer, int)):
                    ordered[axis] = slice(axis_idx, axis_idx + 1, 1)
                elif isinstance(axis_idx, np.ndarray) and not np.issubdtype(
                    axis_idx.dtype, np.bool_
                ):
                    order = np.argsort(axis_idx)
                    ordered[axis] = axis_idx[order]
                    rev_order[axis] = np.argsort(order)
                    order_changed = True

            if order_changed:
                return d[tuple(ordered)][tuple(rev_order)]
        elif isinstance(idx, np.ndarray) and not np.issubdtype(idx.dtype, np.bool_):
            order = np.argsort(idx)
            rev_order = np.argsort(order)
            x = d[idx[order]]
            return x[rev_order]
        return d[idx]


try:
    import h5py

    register_dataset(h5py.Group, h5py.Dataset, Hdf5DenseArray)
except ImportError:
    pass
