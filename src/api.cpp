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


// Sparse Dual Marching Cubes
std::tuple<at::Tensor, at::Tensor> sparse_dual_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d) {
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

    auto [vertices_dev, quads_dev] = _sparse_dual_marching_cubes(
        coords.data_ptr<int>(),
        corners.data_ptr<float>(),
        N,
        iso,
        stream
    );

    // Convert results to Tensors
    at::Tensor verts_tensor = torch::empty({(long)vertices_dev.size(), 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    at::Tensor quads_tensor = torch::empty({(long)quads_dev.size(), 4}, torch::dtype(torch::kInt32).device(coords.device()));

    if (!vertices_dev.empty()) {
        cudaMemcpyAsync(verts_tensor.data_ptr<float>(), vertices_dev.data().get(), vertices_dev.size() * sizeof(V3f), cudaMemcpyDeviceToDevice, stream);
    }
    if (!quads_dev.empty()) {
        cudaMemcpyAsync(quads_tensor.data_ptr<int>(), quads_dev.data().get(), quads_dev.size() * sizeof(Quad), cudaMemcpyDeviceToDevice, stream);
    }

    return {verts_tensor, quads_tensor};
}

std::tuple<at::Tensor, at::Tensor> sparse_dual_marching_cubes_from_points(at::Tensor coords, at::Tensor point_values, double iso_d, double default_value_d) {
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

    auto [vertices_dev, quads_dev] = _sparse_dual_marching_cubes_from_points(
        coords.data_ptr<int>(),
        point_values.data_ptr<float>(),
        N,
        iso,
        default_value,
        stream
    );

    // Convert results to Tensors
    at::Tensor verts_tensor = torch::empty({(long)vertices_dev.size(), 3}, torch::dtype(torch::kFloat32).device(coords.device()));
    at::Tensor quads_tensor = torch::empty({(long)quads_dev.size(), 4}, torch::dtype(torch::kInt32).device(coords.device()));

    if (!vertices_dev.empty()) {
        cudaMemcpyAsync(verts_tensor.data_ptr<float>(), vertices_dev.data().get(), vertices_dev.size() * sizeof(V3f), cudaMemcpyDeviceToDevice, stream);
    }
    if (!quads_dev.empty()) {
        cudaMemcpyAsync(quads_tensor.data_ptr<int>(), quads_dev.data().get(), quads_dev.size() * sizeof(Quad), cudaMemcpyDeviceToDevice, stream);
    }

    return {verts_tensor, quads_tensor};
}


// Mesh Decimation
std::tuple<at::Tensor, at::Tensor> decimate_mesh(at::Tensor vertices, at::Tensor faces, int target_vertex_count) {
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must have shape [N, 3]");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must have shape [F, 3]");
    TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
    TORCH_CHECK(faces.is_contiguous(), "faces must be contiguous");
    TORCH_CHECK(target_vertex_count > 0, "target_vertex_count must be positive");

    int num_vertices = vertices.size(0);
    int num_faces = faces.size(0);

    // Convert to CPU if on CUDA
    at::Tensor vertices_cpu = vertices.cpu().to(torch::kFloat32);
    at::Tensor faces_cpu = faces.cpu().to(torch::kInt32);

    std::vector<V3f> verts_vec(num_vertices);
    std::vector<Tri> faces_vec(num_faces);

    float* v_ptr = vertices_cpu.data_ptr<float>();
    int* f_ptr = faces_cpu.data_ptr<int>();

    for (int i = 0; i < num_vertices; ++i) {
        verts_vec[i].x = v_ptr[i * 3 + 0];
        verts_vec[i].y = v_ptr[i * 3 + 1];
        verts_vec[i].z = v_ptr[i * 3 + 2];
    }

    for (int i = 0; i < num_faces; ++i) {
        faces_vec[i].v0 = f_ptr[i * 3 + 0];
        faces_vec[i].v1 = f_ptr[i * 3 + 1];
        faces_vec[i].v2 = f_ptr[i * 3 + 2];
    }

    // Call CUDA decimation
    decimate_cuda(verts_vec, faces_vec, target_vertex_count);

    // Convert back to tensors
    int new_num_vertices = verts_vec.size();
    int new_num_faces = faces_vec.size();

    at::Tensor new_vertices = torch::empty({new_num_vertices, 3}, torch::dtype(torch::kFloat32));
    at::Tensor new_faces = torch::empty({new_num_faces, 3}, torch::dtype(torch::kInt32));

    float* new_v_ptr = new_vertices.data_ptr<float>();
    int* new_f_ptr = new_faces.data_ptr<int>();

    for (int i = 0; i < new_num_vertices; ++i) {
        new_v_ptr[i * 3 + 0] = verts_vec[i].x;
        new_v_ptr[i * 3 + 1] = verts_vec[i].y;
        new_v_ptr[i * 3 + 2] = verts_vec[i].z;
    }

    for (int i = 0; i < new_num_faces; ++i) {
        new_f_ptr[i * 3 + 0] = faces_vec[i].v0;
        new_f_ptr[i * 3 + 1] = faces_vec[i].v1;
        new_f_ptr[i * 3 + 2] = faces_vec[i].v2;
    }

    // Move to original device if needed
    if (vertices.is_cuda()) {
        new_vertices = new_vertices.to(vertices.device());
        new_faces = new_faces.to(faces.device());
    }

    return {new_vertices, new_faces};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Sparse Marching Cubes
    m.def("sparse_marching_cubes", &sparse_marching_cubes, "Sparse Marching Cubes (original)",
          py::arg("coords"), py::arg("corners"), py::arg("iso"), py::arg("ensure_consistency") = false);
    m.def("sparse_marching_cubes_from_points", &sparse_marching_cubes_from_points, "Sparse Marching Cubes from a grid of points",
          py::arg("coords"), py::arg("point_values"), py::arg("iso"), py::arg("default_value") = 1.0);

    // Sparse Dual Marching Cubes
    m.def("sparse_dual_marching_cubes", &sparse_dual_marching_cubes, "Sparse Dual Marching Cubes",
          py::arg("coords"), py::arg("corners"), py::arg("iso"));
    m.def("sparse_dual_marching_cubes_from_points", &sparse_dual_marching_cubes_from_points, "Sparse Dual Marching Cubes from a grid of points",
          py::arg("coords"), py::arg("point_values"), py::arg("iso"), py::arg("default_value") = 1.0);

    // Mesh Decimation
    m.def("decimate_mesh", &decimate_mesh, "Decimate mesh using QEM edge collapse",
          py::arg("vertices"), py::arg("faces"), py::arg("target_vertex_count"));
}
