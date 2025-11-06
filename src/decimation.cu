#include "api.h" 
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <numeric>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>


#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct Quadricf {
    float a00, a01, a02, a11, a12, a22;
    float b0, b1, b2;
    float c;

    __host__ __device__ Quadricf() {
        a00 = a01 = a02 = a11 = a12 = a22 = 0.f;
        b0 = b1 = b2 = 0.f;
        c = 0.f;
    }
};

// v0 < v1
struct Edge {
    int v0, v1;
    __host__ __device__ bool operator<(const Edge& other) const {
        return (v0 < other.v0) || (v0 == other.v0 && v1 < other.v1);
    }
    __host__ __device__ bool operator==(const Edge& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }
};


// ===================================================================
// Device-Side utility functions
// ===================================================================
__device__ inline V3f operator+(const V3f& a, const V3f& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline V3f operator-(const V3f& a, const V3f& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline V3f operator*(const V3f& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ inline float dot(const V3f& a, const V3f& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline V3f cross(const V3f& a, const V3f& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ inline float squared_norm(const V3f& v) { return dot(v, v); }
__device__ inline float norm(const V3f& v) { return sqrtf(squared_norm(v)); }
__device__ inline V3f normalize(const V3f& v) { float n = norm(v); return n > 1e-6f ? V3f{v.x * (1.f/n), v.y * (1.f/n), v.z * (1.f/n)} : V3f{0.f,0.f,0.f}; }

// Additional geometric utility functions for improved quality checks
__device__ inline float triangle_area(const V3f& p0, const V3f& p1, const V3f& p2) {
    return 0.5f * norm(cross(p1 - p0, p2 - p0));
}

__device__ inline float triangle_min_angle(const V3f& p0, const V3f& p1, const V3f& p2) {
    V3f e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    float l0 = norm(e0), l1 = norm(e1), l2 = norm(e2);
    if (l0 < 1e-8f || l1 < 1e-8f || l2 < 1e-8f) return 0.0f;

    // Use law of cosines to find angles
    float cos_a0 = dot(-e0, e2) / (l0 * l2);
    float cos_a1 = dot(-e1, e0) / (l1 * l0);
    float cos_a2 = dot(-e2, e1) / (l2 * l1);

    // Clamp to avoid numerical issues
    cos_a0 = fminf(1.0f, fmaxf(-1.0f, cos_a0));
    cos_a1 = fminf(1.0f, fmaxf(-1.0f, cos_a1));
    cos_a2 = fminf(1.0f, fmaxf(-1.0f, cos_a2));

    return fminf(acosf(cos_a0), fminf(acosf(cos_a1), acosf(cos_a2)));
}

__device__ inline float triangle_aspect_ratio(const V3f& p0, const V3f& p1, const V3f& p2) {
    V3f e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    float l0 = norm(e0), l1 = norm(e1), l2 = norm(e2);
    float max_edge = fmaxf(l0, fmaxf(l1, l2));
    float area = triangle_area(p0, p1, p2);
    if (area < 1e-8f) return 1e6f; // Very bad aspect ratio
    // Aspect ratio: ratio of circumradius to inradius (normalized)
    return max_edge / (4.0f * area / (l0 + l1 + l2));
}

// Check if a point is on the same side of a plane as the reference point
__device__ inline bool same_side_of_plane(const V3f& point, const V3f& ref, const V3f& plane_p, const V3f& plane_n) {
    float d1 = dot(point - plane_p, plane_n);
    float d2 = dot(ref - plane_p, plane_n);
    return d1 * d2 >= -1e-6f; // Small tolerance for numerical stability
}

// Check if edge (e0, e1) intersects triangle (t0, t1, t2) - using robust method
__device__ inline bool edge_intersects_triangle(const V3f& e0, const V3f& e1,
                                                 const V3f& t0, const V3f& t1, const V3f& t2) {
    // Skip if edge shares vertices with triangle
    float eps = 1e-7f;
    if (squared_norm(e0 - t0) < eps || squared_norm(e0 - t1) < eps || squared_norm(e0 - t2) < eps) return false;
    if (squared_norm(e1 - t0) < eps || squared_norm(e1 - t1) < eps || squared_norm(e1 - t2) < eps) return false;

    V3f tri_normal = cross(t1 - t0, t2 - t0);
    float tri_area_sq = squared_norm(tri_normal);
    if (tri_area_sq < 1e-12f) return false; // Degenerate triangle

    tri_normal = normalize(tri_normal);
    float d = -dot(tri_normal, t0);

    // Check if edge endpoints are on opposite sides of triangle plane
    float dist0 = dot(tri_normal, e0) + d;
    float dist1 = dot(tri_normal, e1) + d;

    if (dist0 * dist1 > 1e-8f) return false; // Both on same side

    // Calculate intersection point
    float t = dist0 / (dist0 - dist1);
    if (t < -1e-6f || t > 1.0f + 1e-6f) return false; // Outside edge

    V3f intersection = e0 + (e1 - e0) * t;

    // Check if intersection point is inside triangle using barycentric coordinates
    V3f v0 = t1 - t0, v1 = t2 - t0, v2 = intersection - t0;
    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);

    float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    return (u >= -1e-6f) && (v >= -1e-6f) && (u + v <= 1.0f + 1e-6f);
}

__device__ inline void atomicAdd(Quadricf* addr, const Quadricf& val) {
    atomicAdd(&addr->a00, val.a00); atomicAdd(&addr->a01, val.a01); atomicAdd(&addr->a02, val.a02);
    atomicAdd(&addr->a11, val.a11); atomicAdd(&addr->a12, val.a12); atomicAdd(&addr->a22, val.a22);
    atomicAdd(&addr->b0, val.b0); atomicAdd(&addr->b1, val.b1); atomicAdd(&addr->b2, val.b2);
    atomicAdd(&addr->c, val.c);
}

// ===================================================================
// CUDA kernel functions
// ===================================================================

__global__ void kernel_calculate_vertex_quadrics(
    const V3f* vertices, const Tri* faces, int num_faces, Quadricf* vertex_quadrics) {

    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) return;

    Tri f = faces[face_idx];
    V3f v0 = vertices[f.v0];
    V3f v1 = vertices[f.v1];
    V3f v2 = vertices[f.v2];

    V3f n = cross(v1 - v0, v2 - v0);
    float area = 0.5f * norm(n);
    if (area <= 1e-8f) return;

    n = normalize(n);
    float d = -dot(n, v0); // Use ax+by+cz+d=0 form

    Quadricf q;
    q.a00 = n.x * n.x; q.a01 = n.x * n.y; q.a02 = n.x * n.z;
    q.a11 = n.y * n.y; q.a12 = n.y * n.z; q.a22 = n.z * n.z;
    q.b0 = n.x * d; q.b1 = n.y * d; q.b2 = n.z * d;
    q.c = d * d;

    // We can remove area weighting if it causes issues, but it's generally good
    q.a00 *= area; q.a01 *= area; q.a02 *= area;
    q.a11 *= area; q.a12 *= area; q.a22 *= area;
    q.b0 *= area; q.b1 *= area; q.b2 *= area;
    q.c *= area;

    atomicAdd(&vertex_quadrics[f.v0], q);
    atomicAdd(&vertex_quadrics[f.v1], q);
    atomicAdd(&vertex_quadrics[f.v2], q);
}


__global__ void kernel_extract_face_edges(const Tri* faces, int num_faces, Edge* all_edges) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) return;
    
    Tri f = faces[face_idx];
    int v0 = f.v0, v1 = f.v1, v2 = f.v2; 

    all_edges[face_idx * 3 + 0] = {min(v0, v1), max(v0, v1)};
    all_edges[face_idx * 3 + 1] = {min(v1, v2), max(v1, v2)};
    all_edges[face_idx * 3 + 2] = {min(v2, v0), max(v2, v0)};
}


__global__ void kernel_calculate_edge_costs(
    const Edge* edges, int num_edges,
    const V3f* vertices, const Quadricf* vertex_quadrics,
    const bool* is_boundary, // Flag for boundary edges
    float* costs, V3f* collapse_pos, 
    const Tri* faces,
    const int* vertex_to_face_map,
    const int* vertex_to_face_offsets) {

    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;
    
    Edge e = edges[edge_idx];
    int v0_idx = e.v0;
    int v1_idx = e.v1;

    Quadricf q0 = vertex_quadrics[v0_idx];
    Quadricf q1 = vertex_quadrics[v1_idx];
    
    Quadricf q;
    q.a00 = q0.a00 + q1.a00; q.a01 = q0.a01 + q1.a01; q.a02 = q0.a02 + q1.a02;
    q.a11 = q0.a11 + q1.a11; q.a12 = q0.a12 + q1.a12; q.a22 = q0.a22 + q1.a22;
    q.b0 = q0.b0 + q1.b0; q.b1 = q0.b1 + q1.b1; q.b2 = q0.b2 + q1.b2;
    q.c = q0.c + q1.c;

    float det = q.a00 * (q.a11 * q.a22 - q.a12 * q.a12) -
                q.a01 * (q.a01 * q.a22 - q.a12 * q.a02) +
                q.a02 * (q.a01 * q.a12 - q.a11 * q.a02);

    V3f p0 = vertices[v0_idx];
    V3f p1 = vertices[v1_idx];
    V3f pos;

    if (fabsf(det) > 1e-8f) {
        float inv_det = 1.0f / det;
        pos.x = -inv_det * ( (q.a11 * q.a22 - q.a12 * q.a12) * q.b0 + (q.a02 * q.a12 - q.a01 * q.a22) * q.b1 + (q.a01 * q.a12 - q.a02 * q.a11) * q.b2 );
        pos.y = -inv_det * ( (q.a02 * q.a12 - q.a01 * q.a22) * q.b0 + (q.a00 * q.a22 - q.a02 * q.a02) * q.b1 + (q.a02 * q.a01 - q.a00 * q.a12) * q.b2 );
        pos.z = -inv_det * ( (q.a01 * q.a12 - q.a02 * q.a11) * q.b0 + (q.a02 * q.a01 - q.a00 * q.a12) * q.b1 + (q.a00 * q.a11 - q.a01 * q.a01) * q.b2 );
    } else {
        pos = (p0 + p1) * 0.5f; // Fallback to midpoint if matrix is singular
    }

    // Sanity check to prevent spikes.
    // If the new position is too far from the original edge, it's likely due to numerical instability.
    // In that case, clamp the position to the edge's midpoint.
    float edge_len_sq = squared_norm(p1 - p0);
    V3f midpoint = (p0 + p1) * 0.5f;
    if (squared_norm(pos - midpoint) > edge_len_sq) {
        pos = midpoint;
    }

    float cost = q.a00*pos.x*pos.x + 2*q.a01*pos.x*pos.y + 2*q.a02*pos.x*pos.z + q.a11*pos.y*pos.y + 2*q.a12*pos.y*pos.z + q.a22*pos.z*pos.z +
                 2*q.b0*pos.x + 2*q.b1*pos.y + 2*q.b2*pos.z + q.c;

    bool causes_flip = false;
    bool causes_degeneracy = false;
    bool causes_self_intersection = false;

    // Quality thresholds
    const float MIN_ANGLE_THRESHOLD = 0.087266f;  // ~5 degrees in radians
    const float MAX_ASPECT_RATIO = 50.0f;         // Maximum allowed aspect ratio
    const float MIN_AREA_THRESHOLD = 1e-10f;      // Minimum triangle area
    const float NORMAL_ANGLE_THRESHOLD = 0.9848f; // cos(10 degrees) for normal change

    // Iterate over one-ring neighborhood of v0
    for (int i = vertex_to_face_offsets[v0_idx]; i < vertex_to_face_offsets[v0_idx+1]; ++i) {
        int face_idx = vertex_to_face_map[i];
        Tri f = faces[face_idx];

        // This face will be removed by the collapse, so no need to check it.
        if ((f.v0 == v0_idx || f.v1 == v0_idx || f.v2 == v0_idx) &&
            (f.v0 == v1_idx || f.v1 == v1_idx || f.v2 == v1_idx)) {
            continue;
        }

        int p_idx = f.v0, q_idx = f.v1, r_idx = f.v2;
        V3f p = vertices[p_idx], q_v = vertices[q_idx], r = vertices[r_idx];

        // Store original triangle
        V3f old_p = p, old_q = q_v, old_r = r;
        V3f old_normal = cross(old_q - old_p, old_r - old_p);
        float old_area_sq = squared_norm(old_normal);

        if (old_area_sq < MIN_AREA_THRESHOLD) continue; // Skip already degenerate faces

        old_normal = normalize(old_normal);

        // Simulate the new positions
        if (p_idx == v0_idx || p_idx == v1_idx) p = pos;
        if (q_idx == v0_idx || q_idx == v1_idx) q_v = pos;
        if (r_idx == v0_idx || r_idx == v1_idx) r = pos;

        V3f new_normal = cross(q_v - p, r - p);
        float new_area = 0.5f * norm(new_normal);

        // IMPROVED DEGENERATE CHECK: Multiple criteria
        if (new_area < MIN_AREA_THRESHOLD) {
            causes_degeneracy = true;
            break;
        }

        float min_angle = triangle_min_angle(p, q_v, r);
        if (min_angle < MIN_ANGLE_THRESHOLD) {
            causes_degeneracy = true;
            break;
        }

        float aspect_ratio = triangle_aspect_ratio(p, q_v, r);
        if (aspect_ratio > MAX_ASPECT_RATIO) {
            causes_degeneracy = true;
            break;
        }

        // IMPROVED NORMAL FLIP CHECK: Use normalized normals and angle threshold
        new_normal = normalize(new_normal);
        float normal_dot = dot(old_normal, new_normal);
        if (normal_dot < 0.0f) {  // Hard flip
            causes_flip = true;
            break;
        }
        if (normal_dot < NORMAL_ANGLE_THRESHOLD) {  // Too much deviation
            causes_flip = true;
            break;
        }
    }

    // v1 one-ring (FIX: was v0_idx, should be v1_idx)
    if (!causes_flip && !causes_degeneracy) {
        for (int i = vertex_to_face_offsets[v1_idx]; i < vertex_to_face_offsets[v1_idx+1]; ++i) {
            int face_idx = vertex_to_face_map[i];
            Tri f = faces[face_idx];

            // This face will be removed by the collapse, so no need to check it.
            if ((f.v0 == v0_idx || f.v1 == v0_idx || f.v2 == v0_idx) &&
                (f.v0 == v1_idx || f.v1 == v1_idx || f.v2 == v1_idx)) {
                continue;
            }

            int p_idx = f.v0, q_idx = f.v1, r_idx = f.v2;
            V3f p = vertices[p_idx], q_v = vertices[q_idx], r = vertices[r_idx];

            // Store original triangle
            V3f old_p = p, old_q = q_v, old_r = r;
            V3f old_normal = cross(old_q - old_p, old_r - old_p);
            float old_area_sq = squared_norm(old_normal);

            if (old_area_sq < MIN_AREA_THRESHOLD) continue;

            old_normal = normalize(old_normal);

            // Simulate the new positions
            if (p_idx == v0_idx || p_idx == v1_idx) p = pos;
            if (q_idx == v0_idx || q_idx == v1_idx) q_v = pos;
            if (r_idx == v0_idx || r_idx == v1_idx) r = pos;

            V3f new_normal = cross(q_v - p, r - p);
            float new_area = 0.5f * norm(new_normal);

            // IMPROVED DEGENERATE CHECK
            if (new_area < MIN_AREA_THRESHOLD) {
                causes_degeneracy = true;
                break;
            }

            float min_angle = triangle_min_angle(p, q_v, r);
            if (min_angle < MIN_ANGLE_THRESHOLD) {
                causes_degeneracy = true;
                break;
            }

            float aspect_ratio = triangle_aspect_ratio(p, q_v, r);
            if (aspect_ratio > MAX_ASPECT_RATIO) {
                causes_degeneracy = true;
                break;
            }

            // IMPROVED NORMAL FLIP CHECK
            new_normal = normalize(new_normal);
            float normal_dot = dot(old_normal, new_normal);
            if (normal_dot < 0.0f) {
                causes_flip = true;
                break;
            }
            if (normal_dot < NORMAL_ANGLE_THRESHOLD) {
                causes_flip = true;
                break;
            }
        }
    }

    // SELF-INTERSECTION CHECK: Check if collapsed edge creates intersections
    // We check the new edges formed by the collapse position against surrounding triangles
    if (!causes_flip && !causes_degeneracy) {
        // Collect edges from v0 one-ring that will remain after collapse
        for (int i = vertex_to_face_offsets[v0_idx]; i < vertex_to_face_offsets[v0_idx+1] && !causes_self_intersection; ++i) {
            int face_idx = vertex_to_face_map[i];
            Tri f = faces[face_idx];

            // Skip faces that will be removed
            if ((f.v0 == v0_idx || f.v1 == v0_idx || f.v2 == v0_idx) &&
                (f.v0 == v1_idx || f.v1 == v1_idx || f.v2 == v1_idx)) {
                continue;
            }

            // Get the other vertices in this triangle (not v0 or v1)
            int other_vertices[2];
            int count = 0;
            if (f.v0 != v0_idx && f.v0 != v1_idx) other_vertices[count++] = f.v0;
            if (f.v1 != v0_idx && f.v1 != v1_idx) other_vertices[count++] = f.v1;
            if (f.v2 != v0_idx && f.v2 != v1_idx) other_vertices[count++] = f.v2;

            // Check new edges against triangles in the two-ring
            if (count == 2) {
                V3f edge_a = vertices[other_vertices[0]];
                V3f edge_b = vertices[other_vertices[1]];

                // Check against v1 one-ring triangles
                for (int j = vertex_to_face_offsets[v1_idx]; j < vertex_to_face_offsets[v1_idx+1]; ++j) {
                    int check_face_idx = vertex_to_face_map[j];
                    if (check_face_idx == face_idx) continue; // Same face

                    Tri cf = faces[check_face_idx];

                    // Skip if this face will be removed or shares vertices
                    if ((cf.v0 == v0_idx || cf.v1 == v0_idx || cf.v2 == v0_idx) &&
                        (cf.v0 == v1_idx || cf.v1 == v1_idx || cf.v2 == v1_idx)) {
                        continue;
                    }

                    // Skip if they share an edge (adjacent triangles)
                    int shared = 0;
                    if (cf.v0 == other_vertices[0] || cf.v0 == other_vertices[1]) shared++;
                    if (cf.v1 == other_vertices[0] || cf.v1 == other_vertices[1]) shared++;
                    if (cf.v2 == other_vertices[0] || cf.v2 == other_vertices[1]) shared++;
                    if (shared >= 2) continue;

                    V3f t0 = vertices[cf.v0], t1 = vertices[cf.v1], t2 = vertices[cf.v2];

                    // Update positions for vertices being collapsed
                    if (cf.v0 == v0_idx || cf.v0 == v1_idx) t0 = pos;
                    if (cf.v1 == v0_idx || cf.v1 == v1_idx) t1 = pos;
                    if (cf.v2 == v0_idx || cf.v2 == v1_idx) t2 = pos;

                    // Check edge-triangle intersection (edges from pos to other vertices)
                    if (edge_intersects_triangle(pos, edge_a, t0, t1, t2) ||
                        edge_intersects_triangle(pos, edge_b, t0, t1, t2)) {
                        causes_self_intersection = true;
                        break;
                    }
                }
            }
        }
    }

    // HARD REJECTION: Set cost to infinity instead of just adding penalty
    if (is_boundary[edge_idx]) {
        cost += edge_len_sq * 1000.0f;  // High penalty but not rejection for boundary
    }
    if (causes_flip || causes_degeneracy || causes_self_intersection) {
        cost = 1e20f;  // Effectively infinite - reject this collapse
        pos = midpoint; // Fallback position (won't be used due to infinite cost)
    }
    
    costs[edge_idx] = cost;
    collapse_pos[edge_idx] = pos;
}

__global__ void kernel_resolve_remap_table(int* remap_table, int num_vertices) {
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= num_vertices) return;

    int current = v_idx;
    int next = remap_table[current];
    while(current != next) {
        current = next;
        next = remap_table[current];
    }
    remap_table[v_idx] = current;
}


__global__ void kernel_update_faces(Tri* faces, int num_faces, const int* remap_table) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) return;
    
    Tri f = faces[face_idx];
    int v0 = remap_table[f.v0];
    int v1 = remap_table[f.v1];
    int v2 = remap_table[f.v2];

    if (v0 == v1 || v1 == v2 || v2 == v0) {
        faces[face_idx] = {-1, -1, -1};
    } else {
        faces[face_idx] = {v0, v1, v2};
    }
}


__global__ void kernel_update_vertex_positions(
    V3f* vertices, const int* remap_table, const V3f* new_positions, const int* collapse_v0_indices, int num_collapses) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_collapses) return;
    
    // The final destination vertex is remap_table[original_v0]
    int v_idx = remap_table[collapse_v0_indices[i]];
    vertices[v_idx] = new_positions[i];
}


__global__ void kernel_compact_and_remap_faces(
    const Tri* old_faces, int num_valid_faces, Tri* new_faces, const int* new_indices) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_valid_faces) return;

    Tri f = old_faces[face_idx];
    new_faces[face_idx] = {new_indices[f.v0], new_indices[f.v1], new_indices[f.v2]};
}

// Functor for atomic marking
struct MarkActiveVertices {
    int* active_ptr;
    const Tri* faces_ptr;
    MarkActiveVertices(int* ap, const Tri* fp) : active_ptr(ap), faces_ptr(fp) {}
    __device__ void operator()(int idx) const {
        Tri f = faces_ptr[idx];
        atomicExch(&active_ptr[f.v0], 1);
        atomicExch(&active_ptr[f.v1], 1);
        atomicExch(&active_ptr[f.v2], 1);
    }
};

// Functor for compacting vertices
struct CompactVertices {
    const V3f* old_v;
    V3f* new_v;
    const int* active;
    const int* indices;
    CompactVertices(const V3f* ov, V3f* nv, const int* a, const int* i) : old_v(ov), new_v(nv), active(a), indices(i) {}
    __device__ void operator()(int idx) const {
        if (active[idx]) {
            new_v[indices[idx]] = old_v[idx];
        }
    }
};


// ===================================================================
// Host-Side Orchestration
// ===================================================================

void build_vertex_to_face_adjacency(int num_vertices, const std::vector<Tri>& faces,
                                    std::vector<int>& vertex_to_face_map,
                                    std::vector<int>& vertex_to_face_offsets) {
    vertex_to_face_offsets.assign(num_vertices + 1, 0);
    std::vector<std::vector<int>> temp_map(num_vertices);

    for (int i = 0; i < faces.size(); ++i) {
        temp_map[faces[i].v0].push_back(i);
        temp_map[faces[i].v1].push_back(i);
        temp_map[faces[i].v2].push_back(i);
    }

    int total_entries = 0;
    for (int i = 0; i < num_vertices; ++i) {
        vertex_to_face_offsets[i] = total_entries;
        total_entries += temp_map[i].size();
    }
    vertex_to_face_offsets[num_vertices] = total_entries;

    vertex_to_face_map.resize(total_entries);
    for (int i = 0; i < num_vertices; ++i) {
        std::copy(temp_map[i].begin(), temp_map[i].end(), vertex_to_face_map.begin() + vertex_to_face_offsets[i]);
    }
}

void decimate_cuda(std::vector<V3f>& vertices, std::vector<Tri>& faces, int target_vertex_count) {
    int num_vertices = vertices.size();
    int num_faces = faces.size();
    if (num_vertices <= target_vertex_count) return;

    thrust::device_vector<V3f> d_vertices = vertices;
    thrust::device_vector<Tri> d_faces = faces;

    const int MAX_ITERATIONS = 10;  // Allow multiple iterations for progressive decimation
    const int THREADS_PER_BLOCK = 256;

    for (int iter = 0; iter < MAX_ITERATIONS && num_vertices > target_vertex_count; ++iter) {
        printf("Iteration %d: Vertices = %d, Faces = %d\n", iter, num_vertices, num_faces);
        if (num_faces == 0) break;

        // Build and transfer adjacency info for this iteration's mesh state
        std::vector<Tri> h_faces(d_faces.size());
        thrust::copy(d_faces.begin(), d_faces.end(), h_faces.begin());

        std::vector<int> h_v_to_f_map, h_v_to_f_offsets;
        build_vertex_to_face_adjacency(num_vertices, h_faces, h_v_to_f_map, h_v_to_f_offsets);
        
        thrust::device_vector<int> d_v_to_f_map = h_v_to_f_map;
        thrust::device_vector<int> d_v_to_f_offsets = h_v_to_f_offsets;

        // 1. Calculate vertex-wise quadrics
        thrust::device_vector<Quadricf> d_vertex_quadrics(num_vertices);
        CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(d_vertex_quadrics.data()), 0, num_vertices * sizeof(Quadricf)));
        
        int grid_size = (num_faces + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_calculate_vertex_quadrics<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            num_faces,
            thrust::raw_pointer_cast(d_vertex_quadrics.data())
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Extract unique edges and calculate costs
        thrust::device_vector<Edge> d_all_edges(num_faces * 3);
        kernel_extract_face_edges<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_faces.data()),
            num_faces,
            thrust::raw_pointer_cast(d_all_edges.data())
        );
        
        thrust::sort(d_all_edges.begin(), d_all_edges.end());
        
        // Identify boundary edges for preservation
        thrust::device_vector<Edge> d_unique_edges = d_all_edges;
        auto unique_end = thrust::unique(thrust::device, d_unique_edges.begin(), d_unique_edges.end());
        d_unique_edges.resize(unique_end - d_unique_edges.begin());
        int num_unique_edges = d_unique_edges.size();

        thrust::device_vector<int> d_edge_counts(num_unique_edges);
        thrust::reduce_by_key(
            d_all_edges.begin(), d_all_edges.end(),
            thrust::constant_iterator<int>(1),
            d_unique_edges.begin(), d_edge_counts.begin()
        );

        thrust::device_vector<bool> d_is_boundary(num_unique_edges);
        thrust::transform(d_edge_counts.begin(), d_edge_counts.end(), d_is_boundary.begin(), 
            [] __device__ (int count) { return count == 1; });
        
        
        thrust::device_vector<float> d_edge_costs(num_unique_edges);
        thrust::device_vector<V3f> d_collapse_pos(num_unique_edges);

        grid_size = (num_unique_edges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_calculate_edge_costs<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_unique_edges.data()), num_unique_edges,
            thrust::raw_pointer_cast(d_vertices.data()), thrust::raw_pointer_cast(d_vertex_quadrics.data()),
            thrust::raw_pointer_cast(d_is_boundary.data()),
            thrust::raw_pointer_cast(d_edge_costs.data()), thrust::raw_pointer_cast(d_collapse_pos.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            thrust::raw_pointer_cast(d_v_to_f_map.data()),
            thrust::raw_pointer_cast(d_v_to_f_offsets.data())
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 3. Select an independent set of edges to collapse
        thrust::device_vector<int> d_edge_indices(num_unique_edges);
        thrust::sequence(d_edge_indices.begin(), d_edge_indices.end());
        
        thrust::sort_by_key(d_edge_costs.begin(), d_edge_costs.end(), d_edge_indices.begin());

        thrust::host_vector<int> h_sorted_indices = d_edge_indices;
        thrust::host_vector<Edge> h_unique_edges = d_unique_edges;
        thrust::host_vector<V3f> h_collapse_pos = d_collapse_pos;
        
        std::vector<bool> vertex_used(num_vertices, false);
        std::vector<std::pair<int, int>> collapses;
        std::vector<V3f> new_positions_batch;
        std::vector<int> collapse_v0_indices_batch;

        // Control the number of collapses per iteration for stability
        // Collapse at most 30% of vertices per iteration to maintain quality
        int max_collapses_per_iter = std::max(1, (int)(num_vertices * 0.3));
        int num_collapses_this_iteration = std::min(max_collapses_per_iter, num_vertices - target_vertex_count);
        if (num_collapses_this_iteration <= 0 && num_vertices > target_vertex_count) {
             num_collapses_this_iteration = 1; 
        } else if (num_collapses_this_iteration <= 0) {
            break; 
        }
        
        for (int i = 0; i < num_unique_edges && collapses.size() < num_collapses_this_iteration; ++i) {
            int original_edge_idx = h_sorted_indices[i];
            Edge e = h_unique_edges[original_edge_idx];
            if (!vertex_used[e.v0] && !vertex_used[e.v1]) {
                vertex_used[e.v0] = true;
                vertex_used[e.v1] = true;
                collapses.push_back({e.v1, e.v0}); // {v_to_remove, v_to_keep}
                new_positions_batch.push_back(h_collapse_pos[original_edge_idx]);
                collapse_v0_indices_batch.push_back(e.v0);
            }
        }
        
        if (collapses.empty()) break;
        int num_collapses = collapses.size();
        
        // 4. Perform edge collapses
        thrust::device_vector<int> d_remap_table(num_vertices);
        thrust::sequence(d_remap_table.begin(), d_remap_table.end());

        thrust::host_vector<int> h_remap_table = d_remap_table;
        for(const auto& p : collapses) { h_remap_table[p.first] = p.second; }
        d_remap_table = h_remap_table;

        grid_size = (num_vertices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_resolve_remap_table<<<grid_size, THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_remap_table.data()), num_vertices);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        thrust::device_vector<int> d_collapse_v0_indices = collapse_v0_indices_batch;
        thrust::device_vector<V3f> d_new_positions_batch = new_positions_batch;
        grid_size = (num_collapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_update_vertex_positions<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_remap_table.data()),
            thrust::raw_pointer_cast(d_new_positions_batch.data()),
            thrust::raw_pointer_cast(d_collapse_v0_indices.data()),
            num_collapses
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        
        grid_size = (num_faces + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_update_faces<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_faces.data()), num_faces,
            thrust::raw_pointer_cast(d_remap_table.data())
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 5. Compact mesh (remove dead faces and unused vertices)
        thrust::device_vector<Tri> d_valid_faces(num_faces);
        auto valid_faces_end = thrust::copy_if(d_faces.begin(), d_faces.end(), d_valid_faces.begin(), 
            [] __device__ (const Tri& f) { return f.v0 != -1; });
        int num_valid_faces = valid_faces_end - d_valid_faces.begin();
        d_valid_faces.resize(num_valid_faces);
        
        if (num_valid_faces == 0) {
            num_vertices = 0;
            num_faces = 0;
            break;
        }

        thrust::device_vector<int> d_active_vertices(num_vertices, 0);
        
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_valid_faces),
            MarkActiveVertices(thrust::raw_pointer_cast(d_active_vertices.data()), thrust::raw_pointer_cast(d_valid_faces.data()))
        );
        
        thrust::device_vector<int> d_new_indices(num_vertices);
        thrust::exclusive_scan(d_active_vertices.begin(), d_active_vertices.end(), d_new_indices.begin());
        
        // The new vertex count is the sum of all elements in d_active_vertices.
        // It can be calculated from the end of the exclusive scan plus the last element of the input.
        thrust::host_vector<int> h_last_active = d_active_vertices;
        thrust::host_vector<int> h_last_new_index = d_new_indices;
        int new_num_vertices = h_last_new_index.back() + h_last_active.back();

        thrust::device_vector<V3f> d_new_vertices(new_num_vertices);
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_vertices),
            CompactVertices(
                thrust::raw_pointer_cast(d_vertices.data()),
                thrust::raw_pointer_cast(d_new_vertices.data()),
                thrust::raw_pointer_cast(d_active_vertices.data()),
                thrust::raw_pointer_cast(d_new_indices.data())
            )
        );

        thrust::device_vector<Tri> d_compacted_faces(num_valid_faces);
        grid_size = (num_valid_faces + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_compact_and_remap_faces<<<grid_size, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_valid_faces.data()), num_valid_faces,
            thrust::raw_pointer_cast(d_compacted_faces.data()),
            thrust::raw_pointer_cast(d_new_indices.data())
        );

        d_vertices.swap(d_new_vertices);
        d_faces.swap(d_compacted_faces);
        num_vertices = new_num_vertices;
        num_faces = num_valid_faces;
    }
    
    vertices.resize(num_vertices);
    faces.resize(num_faces);
    thrust::copy(d_vertices.begin(), d_vertices.end(), vertices.begin());
    thrust::copy(d_faces.begin(), d_faces.end(), faces.begin());
}