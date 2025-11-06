# Sparse CUDA Marching Cubes for PyTorch

This repository contains a high-performance PyTorch extension for performing the Marching Cubes and Dual Marching Cube on sparse voxel grids using CUDA. 
The implementation is written in CUDA C++ and exposed to Python via PyTorch's C++ extension API, allowing seamless integration into existing PyTorch-based workflows.

Core algorithm is derived from amazing [cubvh](https://github.com/ashawkey/cubvh) from [ashakey](https://github.com/ashawkey) and [pdmc](https://github.com/seonghunn/pdmc).

## Features

- **Sparse Voxel:** Operates on explicitly defined voxel coordinates (`[M, 3]`) and their corresponding corner values (`[M, 8]`) or center values (`[N, 3] - [N, 1]`), avoiding the need for dense 3D grids.
- **Sparse Marching Cube:** generates trimesh for given sparse voxel sdfs.
- **Sparse Dual Marching Cube:** generates quadmesh for given sparse voxel sdfs.
- **CUDA QEM Decimation** w/ preventing degenerate faces and normal fliping (experimental).
- **CUDA-Accelerated:** Leverages the massive parallelism of NVIDIA GPUs to process thousands of voxels simultaneously.

![](result.png)

- Left: **Sparse MC** vs. Right: **Sparse DMC** results

## Requirements

- [PyTorch](https://pytorch.org/) (with CUDA support)
- A C++17 compatible compiler (e.g., GCC, Clang, MSVC)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (matching the version used by your PyTorch installation)

## Installation

You can install the package by cloning this repository and running the `setup.py` script.

```bash
# Clone the repository
git clone https://github.com/hwanhuh/spcumc.git
cd spcumc

# Install the package
pip install . 

# clean existing build and re-install in editable mode
rm -rf *.egg-info build && pip install -e .
```

## Usage

### Sparse Marching Cube

For scenarios where the input scalar field is defined at grid points rather than as pre-computed corners for each voxel, the `sparse_marching_cubes_from_points` function provides a convenient and high-performance solution.

This function takes a set of `[N, 3]` integer coordinates and a corresponding `[N]` tensor of scalar values at those points. 
It then uses a high-performance CUDA backend to automatically gather the 8 corner values required for each voxel. 
If a corner does not exist in the provided point set, a default value is used.

This approach is ideal for sparse SDF representations where you have a list of points and their SDF values.

```python
import torch
import spcumc 

# coords, sdfs: [N, 3], [N] cuda tensor
verts, faces = spcumc.sparse_marching_cubes_from_points(
    coords, 
    sdfs, 
    iso_level, 
    default_value=1.0
)

print(f"Generated mesh with {verts.shape[0]} vertices and {faces.shape[0]} triangles.")
```

### Sparse Dual Marching Cube 
```python
import torch
import spcumc 

# coords, sdfs: [N, 3], [N] cuda tensor
verts, faces = spcumc.sparse_dual_marching_cubes_from_points(
    coords, 
    sdfs, 
    iso_level, 
    default_value=1.0
)

print(f"Generated mesh with {verts.shape[0]} vertices and {faces.shape[0]} quads.")
```

### Decimation 

- supports only tri faces

```python
vertices, faces = spcumc.decimate_mesh(vertices, faces, target_vertex_count=10000)
print(f"Generated mesh with {vertices.shape[0]} vertices and {faces.shape[0]} triangles.")
```