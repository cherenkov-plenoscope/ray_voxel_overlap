from scipy.sparse import coo_matrix
from ._cython_overlap import estimate_overlap_of_ray_with_voxels
import numpy as np
import array


def estimate_system_matrix(
    supports,
    directions,
    x_bin_edges,
    y_bin_edges,
    z_bin_edges,
    order="C",
):
    """
    Returns a tomographic system-matrix.
    Along the rows are the voxels, and along the columns are the rays.
    Each matrix-element represents the overlap in euclidean distance of the
    corresponding ray with the voxel. Here each ray represents a single
    read-ot-channel.

    As the system-matrix is usually very sparse, we construct and return a
    scipy.sparse matrix.

    Parameters
    ----------
    supports        [N x 3] 2D array, N 3D support-vectors of the rays.

    directions      [N x 3] 2D array, N 3D direction-vectors of the rays.

    x_bin_edges     1D array of bin-edge-positions along x-axis.

    y_bin_edges     1D array of bin-edge-positions along y-axis.

    z_bin_edges     1D array of bin-edge-positions along z-axis.

    order           Default: 'C'. The ordering of the flat voxel indices.


    Returns
    -------
    system_matrix   2D sparse matrix [num voxels x num rays], Contains
                    the eucledian overlaps.
    """
    assert supports.shape[0] == directions.shape[0], (
        "number of support vectors ({0:d}) must match "
        + "the number of direction vectors ({1:d})".format(
            supports.shape[0], directions.shape[0]
        )
    )
    x_num = x_bin_edges.shape[0] - 1
    y_num = y_bin_edges.shape[0] - 1
    z_num = z_bin_edges.shape[0] - 1
    rays_num = supports.shape[0]

    ray_voxel_overlap = array.array("f")
    ray_indicies = array.array("L")
    voxel_indicies = array.array("L")

    for ray_idx in range(rays_num):
        ov = estimate_overlap_of_ray_with_voxels(
            support=supports[ray_idx],
            direction=directions[ray_idx],
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            z_bin_edges=z_bin_edges,
        )

        voxel_idxs = np.ravel_multi_index(
            np.array([ov["x"], ov["y"], ov["z"]]),
            dims=(x_num, y_num, z_num),
            order=order,
        )

        ray_idxs = ray_idx * np.ones(voxel_idxs.shape[0], dtype=np.uint32)

        ray_voxel_overlap.extend(ov["overlap"])
        ray_indicies.extend(ray_idxs)
        voxel_indicies.extend(voxel_idxs)

    sys_matrix = coo_matrix(
        (ray_voxel_overlap, (voxel_indicies, ray_indicies)),
        shape=(x_num * y_num * z_num, rays_num),
        dtype=np.float32,
    )

    return sys_matrix.tocsr()


def estimate_overlap_of_ray_bundle_with_voxels(
    supports,
    directions,
    x_bin_edges,
    y_bin_edges,
    z_bin_edges,
    order="C",
):
    """
    A ray-bundle represents rays treated as a single read-out-channel.
    When a single ray is not accurate enough to describe the geometry
    of a single read-out-channel, you can represent the  geometry of
    a single read-out-channel using a bundle of multiple rays.

    Parameters
    ----------
    supports        [N x 3] 2D array, N 3D support-vectors in the ray-bundle.

    directions      [N x 3] 2D array, N 3D direction-vectors in the ray-bundle.

    x_bin_edges     1D array of bin-edge-positions along x-axis.

    y_bin_edges     1D array of bin-edge-positions along y-axis.

    z_bin_edges     1D array of bin-edge-positions along z-axis.

    order           Default: 'C'. The ordering of the flat voxel-indices.

    Returns
    -------

    ray_voxel_overlaps      1D array. Average eucledian overlap of the
                            bundle and a voxel.

    voxel_indicies          1D array. Indices of the voxels.
    """
    num_rays_in_bundle = supports.shape[0]
    assert num_rays_in_bundle == directions.shape[0], (
        "number of support vectors ({0:d}) must match "
        + "the number of direction vectors ({1:d})".format(
            supports.shape[0], directions.shape[0]
        )
    )

    x_num = x_bin_edges.shape[0] - 1
    y_num = y_bin_edges.shape[0] - 1
    z_num = z_bin_edges.shape[0] - 1

    overlap_dict = {}
    for ray_idx in range(num_rays_in_bundle):
        sample_ov = estimate_overlap_of_ray_with_voxels(
            support=supports[ray_idx],
            direction=directions[ray_idx],
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            z_bin_edges=z_bin_edges,
        )

        sample_voxel_idxs = np.ravel_multi_index(
            np.array([sample_ov["x"], sample_ov["y"], sample_ov["z"]]),
            dims=(x_num, y_num, z_num),
            order=order,
        )

        for vi in range(len(sample_voxel_idxs)):
            voxel_idx = sample_voxel_idxs[vi]
            if voxel_idx in overlap_dict.keys():
                overlap_dict[voxel_idx] += sample_ov["overlap"][vi]
            else:
                overlap_dict[voxel_idx] = sample_ov["overlap"][vi]

    ray_voxel_overlaps = []
    voxel_indicies = []
    for voxel_idx in overlap_dict:
        voxel_indicies.append(voxel_idx)
        ray_voxel_overlaps.append(overlap_dict[voxel_idx] / num_rays_in_bundle)

    return np.array(ray_voxel_overlaps), np.array(voxel_indicies)
