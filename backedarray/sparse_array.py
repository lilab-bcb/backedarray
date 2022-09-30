"""\
This module implements on disk sparse datasets.

This code is based on and uses the conventions of h5sparse_ by `Appier Inc.`_.
See the copyright and license note in this directory source code.

.. _h5sparse: https://github.com/appier/h5sparse
.. _Appier Inc.: https://www.appier.com/
"""
from typing import NamedTuple, Tuple, Union

import numpy as np
import scipy.sparse

from backedarray.types import sparse_array_type


def copy_array(mtx):
    format_class = get_array_format(mtx.format).in_memory_type
    mtx_copy = format_class(mtx.shape, dtype=mtx.dtype)
    mtx_copy.data = mtx.data[...]
    mtx_copy.indices = mtx.indices[...]
    mtx_copy.indptr = mtx.indptr[...]
    return mtx_copy


class BackedSparseArray:
    """
    Class for backed sparse arrays.
    """

    def __init__(self, group):
        format_class = get_array_format(get_backed_format(group)).in_memory_type
        mtx = format_class(get_backed_shape(group), dtype=get_backed_dtype(group))
        mtx.data = group["data"]
        mtx.indices = group["indices"]
        mtx.indptr = group["indptr"][:]
        self.mtx = mtx
        self.group = group  # for appending

    @property
    def ndim(self):
        return self.mtx.ndim

    @property
    def nnz(self):
        return self.mtx.nnz

    @property
    def value(self):
        return self[()]

    @property
    def dtype(self) -> np.dtype:
        return self.mtx.dtype

    @property
    def indices(self) -> np.ndarray:
        return self.mtx.indices

    @property
    def indptr(self) -> np.ndarray:
        return self.mtx.indptr

    @property
    def data(self) -> np.ndarray:
        return self.mtx.data

    @property
    def format(self) -> str:
        return self.mtx.format

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mtx.shape

    def __getitem__(self, key):
        if (
            key is Ellipsis
            or (isinstance(key, tuple) and len(key) == 0)
            or (isinstance(key, slice) and key == slice(None))
        ):
            return self.copy()
        # handle 1-d slices, avoiding: we have not yet implemented 1D sparse slices; please index using explicit indices, e.g. `x[:, [0]]`
        # uncomment below if we switch from matrices to arrays
        # if isinstance(key, (np.integer, int)):
        #     return self.mtx[[key]]
        # if isinstance(key, tuple) and len(key) == 2:
        #     if isinstance(key[0], (np.integer, int)) and not isinstance(key[1], (np.integer, int)):
        #         key = [key[0]], key[1]
        #
        #         value = self.mtx[key]
        #         return value
        #     elif isinstance(key[1], (np.integer, int)) and not isinstance(
        #         key[0], (np.integer, int)
        #     ):
        #         key = key[0], [key[1]]
        #         value = self.mtx[key]
        #         return value
        return self.mtx[key]

    def copy(self):
        """Returns an in-memory copy of this array.
        No data/indices will be shared between the returned value and current
        array.
        """
        return copy_array(self.mtx)

    def append(self, sparse_array: Union["BackedSparseArray", sparse_array_type]):
        shape = self.shape
        backed_format = self.format
        group = self.group

        if not scipy.sparse.issparse(sparse_array) and not isinstance(
            sparse_array, BackedSparseArray
        ):
            raise NotImplementedError(
                "Currently, only sparse arrays of equivalent format can be " "appended"
            )
        if backed_format not in {"csr", "csc"}:
            raise NotImplementedError(
                f"The append method for format {backed_format} " f"is not implemented."
            )
        if backed_format != sparse_array.format:
            raise ValueError(
                f"Arrays must have same format. Currently are "
                f"{backed_format!r} and {sparse_array.format!r}"
            )

        # shape
        if backed_format == "csr":
            assert (
                shape[1] == sparse_array.shape[1]
            ), "CSR arrays must have same size of dimension 1 to be appended."
            new_shape = (shape[0] + sparse_array.shape[0], shape[1])
        elif backed_format == "csc":
            assert (
                shape[0] == sparse_array.shape[0]
            ), "CSC arrays must have same size of dimension 0 to be appended."
            new_shape = (shape[0], shape[1] + sparse_array.shape[1])

        # data
        data = self.group["data"]
        orig_data_size = data.shape[0]
        data.resize((orig_data_size + sparse_array.data.shape[0],))
        data[orig_data_size:] = sparse_array.data

        # indptr
        indptr = self.group["indptr"]
        orig_data_size = indptr.shape[0]
        append_offset = indptr[-1]

        indptr.resize((orig_data_size + sparse_array.indptr.shape[0] - 1,))
        indptr[orig_data_size:] = sparse_array.indptr[1:].astype(np.int64) + append_offset

        # indices
        indices = self.group["indices"]
        orig_data_size = indices.shape[0]
        indices.resize((orig_data_size + sparse_array.indices.shape[0],))
        indices[orig_data_size:] = sparse_array.indices[...]
        if "h5sparse_shape" in self.group.attrs:
            del group.attrs["h5sparse_shape"]
        group.attrs["shape"] = new_shape

        format_class = get_array_format(self.format).in_memory_type
        mtx = format_class(new_shape, dtype=self.dtype)
        mtx.data = group["data"]
        mtx.indices = group["indices"]
        mtx.indptr = group["indptr"][:]
        self.mtx = mtx


class ArrayFormat(NamedTuple):
    format_str: str
    in_memory_type: sparse_array_type


FORMATS = [
    # ArrayFormat("csr", scipy.sparse.csr_array),
    # ArrayFormat("csc", scipy.sparse.csc_array),
    ArrayFormat("csr", scipy.sparse.csr_matrix),
    ArrayFormat("csc", scipy.sparse.csc_matrix),
]


def get_backed_dtype(group) -> np.dtype:
    return group["data"].dtype


def get_backed_shape(group) -> Tuple[int, int]:
    shape = group.attrs.get("h5sparse_shape")
    return tuple(group.attrs["shape"] if shape is None else shape)


def get_backed_format(group) -> str:
    if "h5sparse_format" in group.attrs:
        return group.attrs["h5sparse_format"]
    else:
        # Should this be an extra field?
        return group.attrs["encoding-type"].replace("_matrix", "")


def get_array_format(format_str: str) -> ArrayFormat:
    for f in FORMATS:
        if format_str == f.format_str:
            return f
    raise ValueError(f"Format string {format_str} is not supported.")
