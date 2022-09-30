from typing import Union

import scipy.sparse


backed_group_type = Union["zarr.hierarchy.Group", "h5py.Group"]  # noqa: F821
backed_array_type = Union["zarr.core.Array", "h5py.Dataset"]  # noqa: F821
node_type = Union[backed_group_type, backed_array_type]
sparse_array_type = Union[scipy.sparse._arrays._sparray, scipy.sparse.spmatrix]
