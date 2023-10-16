import setuptools
import os
import Cython
from Cython import Build as _
import numpy


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("ray_voxel_overlap", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


extensions = [
    setuptools.Extension(
        name="ray_voxel_overlap._cython_overlap",
        sources=[
            os.path.join(
                "ray_voxel_overlap", "_cython_overlap_implementation.pyx"
            ),
            os.path.join("ray_voxel_overlap", "_c_overlap_implementation.c"),
        ],
        language="c",
        include_dirs=[numpy.get_include()],
    )
]


setuptools.setup(
    name="ray_voxel_overlap",
    version=version,
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    description="Estimate the tomographic system-matrix for rays in voxels.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/ray_voxel_overlap",
    packages=["ray_voxel_overlap"],
    install_requires=["scipy"],
    package_data={
        "ray_voxel_overlap": [
            os.path.join("_c_overlap_implementation.c"),
            os.path.join("_cython_overlap_implementation.pyx"),
        ]
    },
    ext_modules=Cython.Build.cythonize(extensions, language_level=3),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
