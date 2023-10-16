"""
Author: Sebastian A. Mueller

Supported by:
    ETH-Zurich, Switzerland
    MPI-Heidelberg, Germany

Estimate the eucledian overlap of rays and volume-cells (voxels).
Estimate the tomographic system-matrix for your detector.
"""
from .version import __version__
from . import _cython_overlap
from . import _py_overlap
from ._cython_overlap import estimate_overlap_of_ray_with_voxels
from ._system_matrix import estimate_system_matrix
from ._system_matrix import estimate_overlap_of_ray_bundle_with_voxels
