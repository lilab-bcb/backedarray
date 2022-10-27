==============
backedarray
==============

Sparse csc and csr arrays backed by on-disk storage in Zarr_ or HDF5_.
Allows accessing slices of larger than memory arrays.
Inspired by h5sparse_ and anndata_.

Installation
------------

.. code:: ipython3

    pip install backedarray
    
Examples
--------

.. code:: ipython3

    import backedarray as ba
    import scipy.sparse
    import numpy as np
    import h5py
    import zarr



Create Dataset
==============

.. code:: ipython3

    csr_matrix = scipy.sparse.random(100, 50, format="csr", density=0.2)
    dense_array = csr_matrix.toarray()

HDF5 Backend
------------

.. code:: ipython3

    # Write sparse matrix in csc or csr format to hdf5 file
    h5_csr_path = 'csr.h5'
    with h5py.File(h5_csr_path, "w") as f:
        ba.write_sparse(f.create_group("X"), csr_matrix)

.. code:: ipython3

    # Write 2-d numpy array to hdf5
    h5_dense_path = 'dense.h5'
    with h5py.File(h5_dense_path, "w") as f:
        f["X"] = dense_array

Zarr Backend
------------

.. code:: ipython3

    # Write sparse matrix in csc or csr format to zarr file
    zarr_csr_path = 'csr.zarr'
    with zarr.open(zarr_csr_path, mode="w") as f:
        ba.write_sparse(f.create_group("X"), csr_matrix)

.. code:: ipython3

     # Write 2-d numpy array to zarr format
    zarr_dense_path = 'dense.zarr'
    with zarr.open(zarr_dense_path, mode="w") as f:
        f["X"] = dense_array

Read Dataset
============

HDF5 Backend
------------

.. code:: ipython3

    h5_csr_file = h5py.File(h5_csr_path, "r")
    h5_csr_disk = ba.open(h5_csr_file["X"])
    h5_dense_file =  h5py.File(h5_dense_path, "r")
    h5_dense_disk = ba.open(h5_dense_file["X"])

Zarr Backend
------------

.. code:: ipython3

    zarr_csr_disk = ba.open(zarr.open(zarr_csr_path)["X"])
    zarr_dense_disk = ba.open(zarr.open(zarr_dense_path)["X"])

Numpy Style Indexing
====================

.. code:: ipython3

    zarr_csr_disk[1:3].toarray()




.. parsed-literal::

    array([[0.        , 0.25620103, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.57643237, 0.7628611 , 0.        , 0.        ,
            0.        , 0.99872378, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.82040632, 0.        ,
            0.09788999, 0.        , 0.        , 0.67186548, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.24171919, 0.        , 0.        ,
            0.        , 0.        , 0.5893689 , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.1650544 ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.98852861, 0.        , 0.01475572,
            0.        , 0.82875194, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.28405987, 0.        , 0.        , 0.72342298,
            0.        , 0.        , 0.        , 0.12985154, 0.        ]])



.. code:: ipython3

    zarr_dense_disk[-2:]




.. parsed-literal::

    array([[0.51141143, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.87214978,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.95867897, 0.        , 0.00825137,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.29541905, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.68913921, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.87239577, 0.        , 0.93164802, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.04102313, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.81888661, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.18858683, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.83726992, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.60594181,
            0.61483901, 0.        , 0.        , 0.37080615, 0.62691013]])



.. code:: ipython3

    h5_csr_disk[2:].toarray()




.. parsed-literal::

    array([[0.        , 0.        , 0.        , ..., 0.        , 0.12985154,
            0.        ],
           [0.        , 0.        , 0.56872386, ..., 0.        , 0.        ,
            0.36926708],
           [0.        , 0.        , 0.75702799, ..., 0.97589322, 0.        ,
            0.34865313],
           ...,
           [0.        , 0.14634835, 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.51141143, 0.        , 0.        , ..., 0.93164802, 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.37080615,
            0.62691013]])



.. code:: ipython3

    h5_csr_disk[...].toarray()




.. parsed-literal::

    array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.25620103, 0.        , ..., 0.5893689 , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.12985154,
            0.        ],
           ...,
           [0.        , 0.14634835, 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.51141143, 0.        , 0.        , ..., 0.93164802, 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.37080615,
            0.62691013]])



.. code:: ipython3

    h5_dense_disk[:2]




.. parsed-literal::

    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.71493443, 0.20460768, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.68284516,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.93012152, 0.        , 0.        , 0.2165738 , 0.        ,
            0.        , 0.        , 0.93954512, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.1808206 , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.25620103, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.57643237, 0.7628611 , 0.        , 0.        ,
            0.        , 0.99872378, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.82040632, 0.        ,
            0.09788999, 0.        , 0.        , 0.67186548, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.24171919, 0.        , 0.        ,
            0.        , 0.        , 0.5893689 , 0.        , 0.        ]])



.. code:: ipython3

    h5_csr_file.close()
    h5_dense_file.close()

Append
======

.. code:: ipython3

    zarr_csr_disk.append(csr_matrix)
    np.testing.assert_array_equal(zarr_csr_disk[...].toarray(), scipy.sparse.vstack((csr_matrix, csr_matrix)).toarray())

Read h5ad files created using `anndata <https://anndata.readthedocs.io/>`__
===========================================================================

.. code:: bash

    %%bash
    if [ ! -f "pbmc3k.h5ad" ]; then
        wget https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad
    fi

.. code:: ipython3

    import anndata.experimental
    with h5py.File('pbmc3k.h5ad', 'r') as f:
        obs = anndata.experimental.read_elem(f['obs'])
        var = anndata.experimental.read_elem(f['var'])
        X = ba.open(f['X'])




.. _Zarr: https://zarr.readthedocs.io/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5
.. _h5sparse: https://github.com/appier/h5sparse
.. _anndata: https://anndata.readthedocs.io
