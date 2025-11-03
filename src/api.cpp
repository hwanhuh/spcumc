#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <torch/extension.h>
#include "api.h"






// C++ interface
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d, bool ensure_consistency) {
    TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(corners.is_cuda(), "corners must be a CUDA tensor");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(corners.is_contiguous(), "corners must be contiguous");
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3, "coords has incorrect shape");
    TORCH_CHECK(corners.dim() == 2 && corners.size(1) == 8, "corners has incorrect shape");
    TORCH_CHECK(coords.size(0) == corners.size(0), "coords and corners must have same number of voxels");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = coords.size(0);
    float iso = (float)iso_d;

    auto result = _sparse_marching_cubes(
        coords.data_ptr<int>(),
        corners.data_ptr<float>(),
        N,
        iso,
        ensure_consistency,
        stream
    );

    // Convert results to Tensors
    at::Tensor verts_tensor = torch::empty({(long)result.first.size(), 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    at::Tensor tris_tensor = torch::empty({(long)result.second.size(), 3}, torch::dtype(torch::kInt32).device(coords.device()));

    if (!result.first.empty()) {
        cudaMemcpyAsync(verts_tensor.data_ptr<float>(), result.first.data().get(), result.first.size() * sizeof(V3f), cudaMemcpyDeviceToDevice, stream);
    }
    if (!result.second.empty()) {
        cudaMemcpyAsync(tris_tensor.data_ptr<int>(), result.second.data().get(), result.second.size() * sizeof(Tri), cudaMemcpyDeviceToDevice, stream);
    }
    
    return {verts_tensor, tris_tensor};
}

std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes_from_points(at::Tensor coords, at::Tensor point_values, double iso_d, double default_value_d) {
    TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(point_values.is_cuda(), "point_values must be a CUDA tensor");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(point_values.is_contiguous(), "point_values must be contiguous");
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3, "coords has incorrect shape");
    TORCH_CHECK(point_values.dim() == 1 || (point_values.dim() == 2 && point_values.size(1) == 1), "point_values has incorrect shape");
    TORCH_CHECK(coords.size(0) == point_values.size(0), "coords and point_values must have same number of elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = coords.size(0);
    float iso = (float)iso_d;
    float default_value = (float)default_value_d;

    auto result = _sparse_marching_cubes_from_points(
        coords.data_ptr<int>(),
        point_values.data_ptr<float>(),
        N,
        iso,
        default_value,
        stream
    );

    // Convert results to Tensors
    at::Tensor verts_tensor = torch::empty({(long)result.first.size(), 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    at::Tensor tris_tensor = torch::empty({(long)result.second.size(), 3}, torch::dtype(torch::kInt32).device(coords.device()));

    if (!result.first.empty()) {
        cudaMemcpyAsync(verts_tensor.data_ptr<float>(), result.first.data().get(), result.first.size() * sizeof(V3f), cudaMemcpyDeviceToDevice, stream);
    }
    if (!result.second.empty()) {
        cudaMemcpyAsync(tris_tensor.data_ptr<int>(), result.second.data().get(), result.second.size() * sizeof(Tri), cudaMemcpyDeviceToDevice, stream);
    }
    
    return {verts_tensor, tris_tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_marching_cubes", &sparse_marching_cubes, "Sparse Marching Cubes (original)",
          py::arg("coords"), py::arg("corners"), py::arg("iso"), py::arg("ensure_consistency") = false);
    m.def("sparse_marching_cubes_from_points", &sparse_marching_cubes_from_points, "Sparse Marching Cubes from a grid of points",
          py::arg("coords"), py::arg("point_values"), py::arg("iso"), py::arg("default_value") = 1.0);
}
