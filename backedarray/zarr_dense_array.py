from backedarray.dense_array import DenseArray
from backedarray.index import Index
from backedarray.utils import register_dataset


class ZarrDenseArray(DenseArray):
    def __init__(self, dataset: "zarr.core.Array"):
        super().__init__(dataset)

    def __getitem__(self, idx: Index):
        d = self.dataset
        return d.oindex[idx]


try:
    import zarr

    register_dataset(zarr.hierarchy.Group, zarr.core.Array, ZarrDenseArray)


except ImportError:
    pass
