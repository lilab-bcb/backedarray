import collections.abc as cabc

import h5py
import zarr
import numpy as np
import pytest
import scipy.sparse

import backedarray
from backedarray import write_sparse
from backedarray.h5f5_dense_array import Hdf5DenseArray
from backedarray.tests.helpers import (
    array_bool_subset,
    array_int_subset,
    array_subset,
    single_subset,
    slice_subset,
)


dense_array = np.random.rand(100, 50)
# csr_mem = scipy.sparse.csr_array(dense_array)
# csc_mem = scipy.sparse.csc_array(dense_array)
csr_mem = scipy.sparse.csr_matrix(dense_array)
csc_mem = scipy.sparse.csc_matrix(dense_array)


def open_file(path, file_format, mode):
    if file_format == "hdf5":
        f = h5py.File(path, mode)
    elif file_format == "zarr":
        f = zarr.open(path, mode=mode)
    else:
        raise ValueError("Unknown file format")
    return f


@pytest.fixture(
    params=[array_subset, slice_subset, array_int_subset, array_bool_subset, single_subset]
)
def subset_func(request):
    return request.param


subset_func2 = subset_func


def assert_array_equal(a, b):
    assert scipy.sparse.issparse(a) == scipy.sparse.issparse(b)
    assert a.shape == b.shape
    a = a.toarray() if scipy.sparse.issparse(a) else a
    b = b.toarray() if scipy.sparse.issparse(b) else b
    np.testing.assert_array_equal(a, b)


@pytest.fixture
def ondisk_equivalent_hdf5(tmp_path):
    csr_path = tmp_path / "csr.h5"
    csc_path = tmp_path / "csc.h5"
    dense_path = tmp_path / "dense.h5"

    with h5py.File(csr_path, "w") as f:
        write_sparse(f.create_group("X"), csr_mem)

    with h5py.File(csc_path, "w") as f:
        write_sparse(f.create_group("X"), csc_mem)

    with h5py.File(dense_path, "w") as f:
        f["X"] = dense_array

    csr_disk = backedarray.open(h5py.File(csr_path, "r")["X"])
    csc_disk = backedarray.open(h5py.File(csc_path, "r")["X"])
    dense_disk = backedarray.open(h5py.File(dense_path, "r")["X"])
    return csr_disk, csc_disk, dense_disk


@pytest.fixture
def ondisk_equivalent_zarr(tmp_path):
    csr_path = tmp_path / "csr.zarr"
    csc_path = tmp_path / "csc.zarr"
    dense_path = tmp_path / "dense.zarr"

    with zarr.open(csr_path, mode="w") as f:
        write_sparse(f.create_group("X"), csr_mem)
    with zarr.open(csc_path, mode="w") as f:
        write_sparse(f.create_group("X"), csc_mem)
    with zarr.open(dense_path, mode="w") as f:
        f["X"] = dense_array

    csr_disk = backedarray.open(zarr.open(csr_path / "X"))
    csc_disk = backedarray.open(zarr.open(csc_path / "X"))
    dense_disk = backedarray.open(zarr.open(dense_path / "X"))
    return csr_disk, csc_disk, dense_disk


def is_fancy_multi_index(subset_idx):
    # note numpy indexing behavior does not work for 2 index arrays array_subset.
    # When accessing a numpy multi-dimensional array with other multi-dimensional arrays of integer type the arrays used for the indices need to have the same shape.
    # Example: dense_matrix[[0,1,2], [2,3]]
    # IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (2,)
    if isinstance(subset_idx, cabc.Iterable) and len(subset_idx) == 2:
        return all(isinstance(x, cabc.Iterable) for x in subset_idx)
    return False


def backed_indexing_slicing(ondisk_equivalent, index_func1, index_func2):
    csr_disk, csc_disk, dense_disk = ondisk_equivalent
    row_index = np.arange(dense_array.shape[0])
    col_index = np.arange(dense_array.shape[1])
    row_index = index_func1(row_index)
    col_index = index_func2(col_index)
    index = (row_index, col_index)
    fancy_multi = is_fancy_multi_index(index)
    # Only one indexing vector or array is currently allowed for fancy indexing
    on_disk_arrays = [dense_disk, csr_disk, csc_disk]
    in_memory_arrays = [dense_array, csr_mem, csc_mem]
    # test slicing both dimensions
    for i in range(len(on_disk_arrays)):
        on_disk_array = on_disk_arrays[i]
        in_mem_array = in_memory_arrays[i]
        if not fancy_multi:
            disk_slice = on_disk_array[index]
            in_mem_array = in_mem_array[index]
        else:
            disk_slice = on_disk_array[index[0]][:, index[1]]
            in_mem_array = in_mem_array[index[0]][:, index[1]]
        # TODO h5py does always match dimensions compared to numpy
        if isinstance(on_disk_array, Hdf5DenseArray) and disk_slice.ndim != in_mem_array.ndim:
            in_mem_array = in_mem_array.flatten()
            disk_slice = disk_slice.flatten()
        assert_array_equal(in_mem_array, disk_slice)

    # test loading all data
    for i in range(len(on_disk_arrays)):
        on_disk_array = on_disk_arrays[i]
        in_mem_array = in_memory_arrays[i]

        assert_array_equal(in_mem_array, on_disk_array[()])
        assert_array_equal(in_mem_array, on_disk_array[:])
        assert_array_equal(in_mem_array, on_disk_array[...])

    # test slicing one dimension
    # note: sparse in memory array errors using ()
    # note: 1-d slices not yet implemented for scipy arrays
    for i in range(len(on_disk_arrays)):
        on_disk_array = on_disk_arrays[i]
        in_mem_array = in_memory_arrays[i]

        in_mem_array_slice1 = in_mem_array[index[0], :]
        in_mem_array_slice2 = in_mem_array[:, index[1]]

        disk_slice1 = on_disk_array[index[0], :]
        disk_slice2 = on_disk_array[:, index[1]]
        if isinstance(on_disk_array, Hdf5DenseArray):  # TODO fix h5py to match numpy
            if in_mem_array_slice1.ndim != disk_slice1.ndim:
                disk_slice1 = disk_slice1.reshape(in_mem_array_slice1.shape)
            if in_mem_array_slice2.ndim != disk_slice2.ndim:
                disk_slice2 = disk_slice2.reshape(in_mem_array_slice2.shape)
        assert_array_equal(in_mem_array_slice1, disk_slice1)
        assert_array_equal(in_mem_array_slice2, disk_slice2)


def test_backed_indexing_zarr(ondisk_equivalent_zarr, subset_func, subset_func2):
    backed_indexing_slicing(ondisk_equivalent_zarr, subset_func, subset_func2)


def test_backed_indexing_hdf5(ondisk_equivalent_hdf5, subset_func, subset_func2):
    backed_indexing_slicing(ondisk_equivalent_hdf5, subset_func, subset_func2)


@pytest.mark.parametrize(
    ["sparse_format", "append_method"],
    [
        pytest.param(scipy.sparse.csr_matrix, scipy.sparse.vstack),
        pytest.param(scipy.sparse.csc_matrix, scipy.sparse.hstack),
    ],
)
@pytest.mark.parametrize("file_format", ["hdf5", "zarr"])
def test_dataset_append_memory(tmp_path, sparse_format, append_method, file_format):
    file_path = tmp_path / str("test." + file_format)
    a = sparse_format(np.random.random((100, 50)))
    b = sparse_format(np.random.random((100, 50)))

    with open_file(file_path, file_format, "a") as f:
        write_sparse(f.create_group("mtx"), a, chunks=(100000,), maxshape=(None,))
        diskmtx = backedarray.open(f["mtx"])

        diskmtx.append(b)
        fromdisk = diskmtx[...]

    frommem = append_method([a, b])

    assert_array_equal(fromdisk, frommem)


@pytest.mark.parametrize(
    ["sparse_format", "append_method"],
    [
        pytest.param(scipy.sparse.csr_matrix, scipy.sparse.vstack),
        pytest.param(scipy.sparse.csc_matrix, scipy.sparse.hstack),
    ],
)
@pytest.mark.parametrize("file_format", ["hdf5", "zarr"])
def test_dataset_append_disk(tmp_path, sparse_format, append_method, file_format):
    file_path = tmp_path / str("test." + file_format)
    a = sparse_format(np.random.random((100, 50)))
    b = sparse_format(np.random.random((100, 50)))
    with open_file(file_path, file_format, "a") as f:
        write_sparse(f.create_group("a"), a, chunks=(100000,), maxshape=(None,))
        write_sparse(f.create_group("b"), b, chunks=(100000,), maxshape=(None,))
        a_disk = backedarray.open(f["a"])
        b_disk = backedarray.open(f["b"])
        a_disk.append(b_disk)
        fromdisk = a_disk[...]

    frommem = append_method([a, b])

    assert_array_equal(fromdisk, frommem)


@pytest.mark.parametrize(
    ["sparse_format", "a_shape", "b_shape"],
    [
        pytest.param("csr", (100, 100), (100, 200)),
        pytest.param("csc", (100, 100), (200, 100)),
    ],
)
@pytest.mark.parametrize("file_format", ["hdf5", "zarr"])
def test_wrong_shape(tmp_path, sparse_format, a_shape, b_shape, file_format):
    file_path = tmp_path / str("test." + file_format)
    a_mem = scipy.sparse.random(*a_shape, format=sparse_format)
    b_mem = scipy.sparse.random(*b_shape, format=sparse_format)

    with open_file(file_path, file_format, "a") as f:
        write_sparse(f.create_group("a"), a_mem)
        write_sparse(f.create_group("b"), b_mem)
        a_disk = backedarray.open(f["a"])
        b_disk = backedarray.open(f["b"])

        with pytest.raises(AssertionError):
            a_disk.append(b_disk)


@pytest.mark.parametrize("file_format", ["hdf5", "zarr"])
def test_wrong_formats(tmp_path, file_format):
    file_path = tmp_path / str("test." + file_format)
    base = scipy.sparse.random(100, 100, format="csr")

    with open_file(file_path, file_format, "a") as f:
        write_sparse(f.create_group("base"), base)
        disk_mtx = backedarray.open(f["base"])
        pre_checks = disk_mtx[...]

        with pytest.raises(ValueError):
            disk_mtx.append(scipy.sparse.random(100, 100, format="csc"))
        with pytest.raises(ValueError):
            disk_mtx.append(scipy.sparse.random(100, 100, format="coo"))
        with pytest.raises(NotImplementedError):
            disk_mtx.append(np.random.random((100, 100)))
        disk_dense = f.create_dataset("dense", data=np.random.random((100, 100)))
        with pytest.raises(NotImplementedError):
            disk_mtx.append(disk_dense)

        post_checks = disk_mtx[...]

    # Check nothing changed
    assert not np.any((pre_checks != post_checks).toarray())
