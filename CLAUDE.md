# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch CUDA extension that implements high-performance sparse marching cubes algorithms for generating meshes from sparse voxel grids. The project uses CUDA C++ for computation and PyTorch's C++ extension API for Python bindings.

Core algorithms derived from [cubvh](https://github.com/ashawkey/cubvh) and [pdmc](https://github.com/seonghunn/pdmc).

## Build and Installation

### Initial Installation
```bash
pip install .
```

### Development Installation (Editable Mode)
```bash
pip install -e .
```

### Clean Rebuild (Required After Modifying CUDA/C++ Code)
```bash
rm -rf *.egg-info build && pip install -e .
```

**Important**: After any changes to CUDA (.cu) or C++ (.cpp/.h) files, you MUST clean and rebuild. Python-only changes don't require rebuilding.

## Architecture

### Three-Layer Architecture

1. **CUDA Kernel Layer** (`src/*.cu`)
   - `spcumc.cu`: Sparse Marching Cubes implementation (generates triangle meshes)
   - `spdmc.cu`: Sparse Dual Marching Cubes implementation (generates quad meshes)
   - `decimation.cu`: QEM-based mesh decimation with degenerate face/normal flip prevention
   - Core compute kernels using Thrust and CUDA primitives

2. **C++ Binding Layer** (`src/api.cpp`, `src/api.h`)
   - PyTorch tensor validation and conversion
   - CUDA stream management
   - pybind11 module definitions
   - Converts between PyTorch tensors and CUDA device vectors
   - Type conversions: handles `float3` (V3f), `Tri`, `Quad` structs

3. **Python Interface Layer** (`api.py`, `spcumc/__init__.py`)
   - High-level Python API with documentation
   - Input validation and tensor preparation (ensure CUDA, contiguous, correct dtype)
   - Exposes: `sparse_marching_cubes`, `sparse_marching_cubes_from_points`,
     `sparse_dual_marching_cubes`, `sparse_dual_marching_cubes_from_points`, `decimate_mesh`

### Key Data Structures

- **V3f**: `float3` - 3D vertex position
- **Tri**: Triangle with 3 vertex indices (v0, v1, v2)
- **Quad**: Quadrilateral with 4 vertex indices (v0, v1, v2, v3)
- **Sparse Representation**: `[N, 3]` integer coords + `[N]` or `[N, 8]` scalar values

### Two Input Modes

1. **From Points**: Takes sparse grid points `[N, 3]` and values `[N]`, automatically gathers 8 corner values per voxel
   - Functions: `*_from_points` variants
   - Use when you have SDF values at grid points
   - `default_value` parameter fills missing corners

2. **From Corners**: Takes voxel coords `[N, 3]` and pre-computed corner values `[N, 8]`
   - Functions: base `sparse_marching_cubes` and `sparse_dual_marching_cubes`
   - Use when corner values are pre-computed
   - `ensure_consistency` parameter can average shared corners

## Requirements

- PyTorch with CUDA support (version must match CUDA toolkit)
- NVIDIA CUDA Toolkit (matching PyTorch's CUDA version)
- C++17 compatible compiler (GCC, Clang, MSVC)
- Thrust library (included with CUDA toolkit)

## Running Demo

```bash
python demo.py
```

This creates a sparse sphere SDF and generates both MC (triangle) and DMC (quad) meshes, saving to OBJ files.

## Common Workflow for Development

1. Modify CUDA/C++ code in `src/`
2. Clean and rebuild: `rm -rf *.egg-info build && pip install -e .`
3. Test changes with `demo.py` or custom script
4. Verify output meshes visually

## Important Implementation Notes

### Tensor Requirements
- All coordinate tensors must be `int32` type
- All value tensors must be `float32` type
- All tensors must be contiguous in memory
- CUDA tensors are preferred (CPU tensors are auto-converted but with overhead)

### Decimation Details
- Only supports triangle meshes (not quads)
- Uses QEM (Quadric Error Metrics) for edge collapse decisions
- Operates on CPU (converts to CPU if input is CUDA, converts back after)

#### Quality Preservation Checks (Improved Algorithm)

The decimation now includes intelligent quality checks that **hard reject** invalid collapses (infinite cost) rather than just penalizing them:

1. **Degenerate Face Prevention**:
   - Minimum triangle area check (1e-10)
   - Minimum angle check (~5 degrees) - prevents needle triangles
   - Maximum aspect ratio check (50:1) - prevents sliver triangles
   - All three criteria must pass for each affected triangle

2. **Normal Flip Prevention**:
   - Uses normalized normals (not area-weighted) for accurate comparison
   - Detects hard flips (dot product < 0)
   - Detects excessive deviation (angle > 10 degrees between old and new normals)
   - Prevents visual artifacts from geometric inversion

3. **Self-Intersection Detection**:
   - Checks edge-triangle intersections in the collapse neighborhood
   - Uses robust barycentric coordinate method
   - Tests new edges formed by collapse position against surrounding triangles
   - Prevents mesh from folding into itself

4. **Progressive Decimation**:
   - Multiple iterations (up to 10) for gradual quality-preserving decimation
   - Collapses at most 30% of vertices per iteration
   - Rebuilds adjacency information each iteration for accurate topology tracking

#### Known Bug Fixed
- Fixed critical bug where v1 one-ring check was using v0's adjacency data (line 338 in decimation.cu)

### Coordinate System
- Uses integer voxel coordinates for sparse representation
- Vertex positions output in floating-point world coordinates
- Marching cubes lookup tables are embedded in CUDA constant memory

## File Structure

```
spcumc/
├── src/              # CUDA/C++ implementation
│   ├── api.cpp       # PyTorch bindings
│   ├── api.h         # C++ interface definitions
│   ├── spcumc.cu     # Sparse Marching Cubes kernels
│   ├── spdmc.cu      # Sparse Dual Marching Cubes kernels
│   └── decimation.cu # Mesh decimation kernels
├── spcumc/           # Python package
│   └── __init__.py   # Exposes C++ functions to Python
├── api.py            # High-level Python API with validation
├── demo.py           # Example usage script
└── setup.py          # Build configuration
```
