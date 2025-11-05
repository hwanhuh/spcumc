#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>
#include <thrust/device_vector.h>

using V3f = float3;
struct Tri { int v0, v1, v2; };
struct Quad { int v0, v1, v2, v3; };

// Sparse Marching Cubes
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d, bool ensure_consistency = false);
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes_from_points(at::Tensor coords, at::Tensor point_values, double iso_d, double default_value);

// Sparse Dual Marching Cubes
std::tuple<at::Tensor, at::Tensor> sparse_dual_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d);
std::tuple<at::Tensor, at::Tensor> sparse_dual_marching_cubes_from_points(at::Tensor coords, at::Tensor point_values, double iso_d, double default_value);

// CUDA forward declarations
std::pair<thrust::device_vector<V3f>, thrust::device_vector<Tri>>
_sparse_marching_cubes(const int* d_coords, const float* d_corners, int N, float iso, bool ensure_consistency, cudaStream_t stream);

std::pair<thrust::device_vector<V3f>, thrust::device_vector<Tri>>
_sparse_marching_cubes_from_points(const int* d_coords, const float* d_point_values, int N, float iso, float default_value, cudaStream_t stream);

std::tuple<thrust::device_vector<V3f>, thrust::device_vector<Quad>>
_sparse_dual_marching_cubes(const int* d_coords, const float* d_corners, int N, float iso, cudaStream_t stream);

std::tuple<thrust::device_vector<V3f>, thrust::device_vector<Quad>>
_sparse_dual_marching_cubes_from_points(const int* d_coords, const float* d_point_values, int N, float iso, float default_value, cudaStream_t stream);

// Mesh Decimation
std::tuple<at::Tensor, at::Tensor> decimate_mesh(at::Tensor vertices, at::Tensor faces, int target_vertex_count);

// Decimation CUDA forward declaration
void decimate_cuda(std::vector<V3f>& vertices, std::vector<Tri>& faces, int target_vertex_count);
