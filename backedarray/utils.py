import scipy

from backedarray.sparse_array import BackedSparseArray
from backedarray.types import backed_group_type, node_type, sparse_array_type


sparse_data_types = set()
dense_data_type_to_impl = {}


def open(node: node_type):
    """
    Open a backed sparse or dense array.

    :param node: A zarr.hierarchy.Group or h5py.Group to open a sparse matrix or a zarr.core.Array or h5py.Dataset to open a dense array.
    :return Backed sparse csc or csr matrix if node is a group, backed dense array otherwise.
    """
    for cls in sparse_data_types:
        if isinstance(node, cls):
            return BackedSparseArray(node)
    for cls in dense_data_type_to_impl:
        if isinstance(node, cls):
            return dense_data_type_to_impl[cls](node)
    raise ValueError("Unknown node type - {}".format(type(node)))


def write_sparse(group: backed_group_type, array: sparse_array_type, **kwargs):
    """
    Writes a sparse csc or crs matrix to disk

    :param group: A zarr.hierarchy.Group or a h5py.Group
    :param array: A sparse csc or csr matrix
    :param kwargs: Arguments passed to group.create_dataset
    """
    if not scipy.sparse.isspmatrix_csr(array) and not scipy.sparse.isspmatrix_csc(array):
        raise ValueError("Only csr and csc matrices supported")
    group.attrs["encoding-type"] = (
        "csr_matrix" if scipy.sparse.isspmatrix_csr(array) else "csc_matrix"
    )
    group.attrs["shape"] = array.shape
    group.create_dataset("data", data=array.data, dtype=array.data.dtype, **kwargs)
    group.create_dataset("indices", data=array.indices, dtype=array.indices.dtype, **kwargs)
    group.create_dataset("indptr", data=array.indptr, dtype=array.indptr.dtype, **kwargs)


def register_dataset(group_type, array_type, dense_impl):
    sparse_data_types.add(group_type)
    dense_data_type_to_impl[array_type] = dense_impl
