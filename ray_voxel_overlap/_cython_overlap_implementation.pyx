import numpy as np
cimport numpy as np
cimport cython


cdef extern double c_ray_box_overlap(
    double * support,
    double * direction,
    double xl, double xu,
    double yl, double yu,
    double zl, double zu)


@cython.boundscheck(False)
@cython.wraparound(False)
def _estimate_ray_box_overlap(
    np.ndarray[double, ndim=1, mode="c"] support not None,
    np.ndarray[double, ndim=1, mode="c"] direction not None,
    double xl, double xu,
    double yl, double yu,
    double zl, double zu
):
    assert support.shape[0] == 3
    assert direction.shape[0] == 3

    cdef double overlap = c_ray_box_overlap(
        & support[0],
        & direction[0],
        xl, xu,
        yl, yu,
        zl, zu)
    return overlap


cdef extern void c_overlap_of_ray_with_voxels(
    double * support,
    double * direction,
    double * x_bin_edges,
    double * y_bin_edges,
    double * z_bin_edges,
    unsigned int * x_range,
    unsigned int * y_range,
    unsigned int * z_range,
    unsigned int * number_overlaps,
    unsigned int * x_idxs,
    unsigned int * y_idxs,
    unsigned int * z_idxs,
    double * overlaps
)


@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_overlap_of_ray_with_voxels(
    support,
    direction,
    x_bin_edges,
    y_bin_edges,
    z_bin_edges,
    x_range=None,
    y_range=None,
    z_range=None
):
    '''
    Returns the voxel indices and overlap distances for one single ray
    (defined by support and direction) with voxels defined by the bin_edges
    in x,y and z.

    support         3D support vector of ray.

    direction       3D direction vector of ray.

    x_bin_edges     voxel bin edge positions in x.

    y_bin_edges     voxel bin edge positions in y.

    z_bin_edges     voxel bin edge positions in z.

    x_range         lower and upper bin indices to truncate voxels in x
                    (optional)
    y_range         lower and upper bin indices to truncate voxels in y
                    (optional)
    z_range         lower and upper bin indices to truncate voxels in z
                    (optional)
    '''
    assert support is not None
    assert len(support) == 3, 'support must have 3 dimension'
    cdef np.ndarray[double, mode="c"] _support = np.ascontiguousarray(
        support,
        dtype=np.float64)

    assert direction is not None
    assert len(direction) == 3, 'direction must have 3 dimension'
    cdef np.ndarray[double, mode="c"] _direction = np.ascontiguousarray(
        direction,
        dtype=np.float64)

    assert x_bin_edges is not None
    assert len(x_bin_edges) >= 2, (
        'Expected len(x_bin_edges) >= 2, ' +
        'but actually len(x_bin_edges) = {0:d}.'.format(len(x_bin_edges)) +
        'Need at least 2 edges to define a bin.')
    cdef np.ndarray[double, mode="c"] _x_bin_edges = np.ascontiguousarray(
        x_bin_edges,
        dtype=np.float64)

    assert y_bin_edges is not None
    assert len(y_bin_edges) >= 2, (
        'Expected len(y_bin_edges) >= 2, ' +
        'but actually len(y_bin_edges) = {0:d}.'.format(len(y_bin_edges)) +
        'Need at least 2 edges to define a bin.')
    cdef np.ndarray[double, mode="c"] _y_bin_edges = np.ascontiguousarray(
        y_bin_edges,
        dtype=np.float64)

    assert z_bin_edges is not None
    assert len(z_bin_edges) >= 2, (
        'Expected len(z_bin_edges) >= 2, ' +
        'but actually len(z_bin_edges) = {0:d}.'.format(len(z_bin_edges)) +
        'Need at least 2 edges to define a bin.')
    cdef np.ndarray[double, mode="c"] _z_bin_edges = np.ascontiguousarray(
        z_bin_edges,
        dtype=np.float64)

    cdef np.ndarray[unsigned int, mode="c"] _x_range = np.ascontiguousarray(
        np.array([0, 0]),
        dtype=np.uint32)
    if x_range is None:
        _x_range[0] = 0
        _x_range[1] = len(_x_bin_edges) - 1
    else:
        _x_range[0] = x_range[0]
        _x_range[1] = x_range[1]

    cdef np.ndarray[unsigned int, mode="c"] _y_range = np.ascontiguousarray(
        np.array([0, 0]),
        dtype=np.uint32)
    if y_range is None:
        _y_range[0] = 0
        _y_range[1] = len(_y_bin_edges) - 1
    else:
        _y_range[0] = y_range[0]
        _y_range[1] = y_range[1]

    cdef np.ndarray[unsigned int, mode="c"] _z_range = np.ascontiguousarray(
        np.array([0, 0]),
        dtype=np.uint32)
    if z_range is None:
        _z_range[0] = 0
        _z_range[1] = len(_z_bin_edges) - 1
    else:
        _z_range[0] = z_range[0]
        _z_range[1] = z_range[1]

    assert _x_range[1] <= _x_bin_edges.shape[0]
    assert _y_range[1] <= _y_bin_edges.shape[0]
    assert _z_range[1] <= _z_bin_edges.shape[0]

    x_range_width = _x_range[1] - _x_range[0]
    y_range_width = _y_range[1] - _y_range[0]
    z_range_width = _z_range[1] - _z_range[0]

    maximal_number_of_overlaps = int(
        4.0*np.sqrt(
           x_range_width**2 +
           y_range_width**2 +
           z_range_width**2))

    cdef np.ndarray[unsigned int, mode="c"] x_idxs = np.zeros(
        maximal_number_of_overlaps,
        dtype=np.uint32)
    cdef np.ndarray[unsigned int, mode="c"] y_idxs = np.zeros(
        maximal_number_of_overlaps,
        dtype=np.uint32)
    cdef np.ndarray[unsigned int, mode="c"] z_idxs = np.zeros(
        maximal_number_of_overlaps,
        dtype=np.uint32)
    cdef np.ndarray[double, mode="c"] overlaps = np.zeros(
        maximal_number_of_overlaps,
        dtype=np.float64)
    cdef unsigned int number_overlaps = 0

    c_overlap_of_ray_with_voxels(
        & _support[0],
        & _direction[0],
        & _x_bin_edges[0],
        & _y_bin_edges[0],
        & _z_bin_edges[0],
        & _x_range[0],
        & _y_range[0],
        & _z_range[0],
        & number_overlaps,
        & x_idxs[0],
        & y_idxs[0],
        & z_idxs[0],
        & overlaps[0])

    return {
        'x': x_idxs[0:number_overlaps],
        'y': y_idxs[0:number_overlaps],
        'z': z_idxs[0:number_overlaps],
        'overlap': overlaps[0:number_overlaps]}
