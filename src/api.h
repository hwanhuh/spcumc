#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>
#include <thrust/device_vector.h>

using V3f = float3;
struct Tri { int v0, v1, v2; };

std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d, bool ensure_consistency = false);
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes_from_points(at::Tensor coords, at::Tensor point_values, double iso_d, double default_value);

// CUDA forward declarations
std::pair<thrust::device_vector<V3f>, thrust::device_vector<Tri>>
_sparse_marching_cubes(const int* d_coords, const float* d_corners, int N, float iso, bool ensure_consistency, cudaStream_t stream);

std::pair<thrust::device_vector<V3f>, thrust::device_vector<Tri>>
_sparse_marching_cubes_from_points(const int* d_coords, const float* d_point_values, int N, float iso, float default_value, cudaStream_t stream);
