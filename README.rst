##########################################
Overlap of a ray and a volume cell (voxel)
##########################################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

Estimate the euclidean overlap passed by a ray within a rectangular volume
cell (voxel).


|img_ray_and_voxel|


For a given, rectangular space partitioning in 3D, and a given ray the
overlap of all voxels with the ray is estimated.
The figure shows a ray and its overlap with voxels.
A brown overlap with voxel ``3``, a red overlap with voxel ``0``, a purple
overlap with voxel ``4``, and a green overlap with voxel ``5``. The ray is
defined by its support and direction vectors. The space-partitioning is
defined by its bin-edges.


*******
Install
*******

.. code-block:: bash

    pip install ray_voxel_overlap


*********
Interface
*********
There is one core function:


.. code-block:: python

    import ray_voxel_overlap
    ray_voxel_overlap.estimate_overlap_of_ray_with_voxels?
    """
    Returns the voxel indices and overlap distances for one single ray
    (defined by support and direction) with voxels defined by the bin_edges
    in x,y and z.

    support         3D support vector of ray.

    direction       3D direction vector of ray.

    x_bin_edges     voxel bin edge positions in x.

    y_bin_edges     voxel bin edge positions in y.

    z_bin_edges     voxel bin edge positions in z.
    """

There are two more functions:

- 2nd ``ray_voxel_overlap.estimate_system_matrix()``

Create a system-matrix using scipy.sparse matrix which can be used for
iterative tomographic reconstructions.

- 3rd ``ray_voxel_overlap.estimate_overlap_of_ray_bundle_with_voxels()``

Average the overlap of multiple rays representing a single read-out-channel.
This is useful when a single ray is not representative enough for the
geometry sensed by a read-out-channel in your tomographic setup, e.g. when
there is a narrow depth-of-field.

*************************
Tomographic system-matrix
*************************

.. code-block:: python

    import numpy as np
    import ray_voxel_overlap as rvo

    np.random.seed(0)

    N_RAYS = 100
    supports = np.array([
        np.random.uniform(-2.5, 2.5, N_RAYS),
        np.random.uniform(-2.5, 2.5, N_RAYS),
        np.zeros(N_RAYS)
    ]).T

    directions = np.array([
        np.random.uniform(-0.3, 0.3, N_RAYS),
        np.random.uniform(-0.3, 0.3, N_RAYS),
        np.ones(N_RAYS)
    ]).T

    norm_directions = np.linalg.norm(directions, axis=1)
    directions[:, 0] /= norm_directions
    directions[:, 1] /= norm_directions
    directions[:, 2] /= norm_directions

    N_X_BINS = 8
    N_Y_BINS = 13
    N_Z_BINS = 7
    system_matrix = rvo.estimate_system_matrix(
        supports=supports,
        directions=directions,
        x_bin_edges=np.linspace(-100., 100., N_X_BINS+1),
        y_bin_edges=np.linspace(-100., 100., N_Y_BINS+1),
        z_bin_edges=np.linspace(0., 200., N_Z_BINS+1),
    )


How it is done
==============
To be fast, the production-code is written in ``C`` and wrapped in ``cython``.
But for development, there is a ``python`` implementation.

Authors
=======
Sebastian A. Mueller,

ETH-Zurich, Switzerland (2014-2019),

MPI-Heidelberg, Germany (2019-)

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/ray_voxel_overlap/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/ray_voxel_overlap/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/ray_voxel_overlap
    :target: https://pypi.org/project/ray_voxel_overlap

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |img_ray_and_voxel| image:: https://github.com/cherenkov-plenoscope/ray_voxel_overlap/blob/main/readme/ray_and_voxel.svg

.. |MITLicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
