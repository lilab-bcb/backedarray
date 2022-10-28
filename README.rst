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

Zarr Backend
------------

.. code:: ipython3

    # Write sparse matrix in csc or csr format to zarr file
    zarr_csr_path = 'csr.zarr'
    with zarr.open(zarr_csr_path, mode="w") as f:
        ba.write_sparse(f.create_group("X"), csr_matrix)

Read Dataset
============

HDF5 Backend
------------

.. code:: ipython3

    h5_csr_file = h5py.File(h5_csr_path, "r")
    h5_csr_disk = ba.open(h5_csr_file["X"])

Zarr Backend
------------

.. code:: ipython3

    zarr_csr_disk = ba.open(zarr.open(zarr_csr_path)["X"])

Numpy Style Indexing
====================

.. code:: ipython3

    zarr_csr_disk[1:3].toarray()




.. parsed-literal::

    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.06275782, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.61030855, 0.46886635, 0.        , 0.11597629,
            0.        , 0.        , 0.        , 0.23471198, 0.        ,
            0.        , 0.        , 0.        , 0.4911036 , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.00851426,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.10065413],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.93545866, 0.        , 0.        , 0.        , 0.        ,
            0.26147665, 0.        , 0.99931215, 0.        , 0.        ,
            0.        , 0.        , 0.18532786, 0.        , 0.69309913,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.32219088, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.14121076, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.70207481, 0.        , 0.        , 0.        , 0.        ]])



.. code:: ipython3

    h5_csr_disk[2:].toarray()




.. parsed-literal::

    array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.89758627, 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.81611075, 0.        , 0.        , ..., 0.82151986, 0.        ,
            0.        ]])



.. code:: ipython3

    h5_csr_disk[...].toarray()




.. parsed-literal::

    array([[0.        , 0.45873864, 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.10065413],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.81611075, 0.        , 0.        , ..., 0.82151986, 0.        ,
            0.        ]])



.. code:: ipython3

    h5_csr_file.close()

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
        wget -q https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad
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
