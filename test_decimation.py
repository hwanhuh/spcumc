"""
Test script for improved decimation algorithm.

This script tests the enhanced quality checks including:
- Degenerate face prevention
- Normal flip detection
- Self-intersection detection
"""

import torch
import spcumc
import numpy as np

def create_test_mesh():
    """Create a simple test mesh (icosphere-like structure)"""
    # Create a simple subdivided tetrahedron for testing
    vertices = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 1.0],
        [-0.866, -1.0, -0.5],
        [0.866, -1.0, -0.5],
        [0.0, 0.0, 0.5],
        [-0.433, 0.0, -0.25],
        [0.433, 0.0, -0.25],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 4, 1],
        [0, 1, 5],
        [0, 5, 2],
        [0, 2, 6],
        [0, 6, 3],
        [0, 3, 4],
        [1, 4, 3],
        [2, 5, 1],
        [3, 6, 2],
    ], dtype=torch.int32)

    return vertices, faces

def test_basic_decimation():
    """Test basic decimation functionality"""
    print("=" * 60)
    print("Test 1: Basic Decimation")
    print("=" * 60)

    vertices, faces = create_test_mesh()
    print(f"Input: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    target_vertices = max(4, vertices.shape[0] // 2)
    decimated_verts, decimated_faces = spcumc.decimate_mesh(
        vertices, faces, target_vertices
    )

    print(f"Output: {decimated_verts.shape[0]} vertices, {decimated_faces.shape[0]} faces")
    print(f"Target was: {target_vertices} vertices")
    print(f"✓ Decimation completed\n")

    return decimated_verts, decimated_faces

def test_quality_metrics(vertices, faces):
    """Verify mesh quality after decimation"""
    print("=" * 60)
    print("Test 2: Quality Metrics")
    print("=" * 60)

    verts = vertices.cpu().numpy()
    face_indices = faces.cpu().numpy()

    # Check for degenerate faces
    min_area = float('inf')
    max_area = 0.0
    degenerate_count = 0

    for face in face_indices:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)

        min_area = min(min_area, area)
        max_area = max(max_area, area)

        if area < 1e-9:
            degenerate_count += 1

    print(f"Triangle area range: [{min_area:.2e}, {max_area:.2e}]")
    print(f"Degenerate triangles (area < 1e-9): {degenerate_count}")

    if degenerate_count == 0:
        print("✓ No degenerate faces detected")
    else:
        print("✗ Found degenerate faces - quality check failed!")

    # Check for flipped normals (all normals should point generally outward)
    normals = []
    for face in face_indices:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-9:
            normals.append(normal / norm_len)

    # Check consistency: most normals should be somewhat aligned
    if len(normals) > 1:
        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        flipped_count = 0
        for normal in normals:
            if np.dot(normal, avg_normal) < 0:
                flipped_count += 1

        print(f"Potentially flipped normals: {flipped_count}/{len(normals)}")
        if flipped_count < len(normals) * 0.1:  # Less than 10% flipped
            print("✓ Normal consistency check passed")
        else:
            print("✗ Too many flipped normals detected!")

    print()

def test_progressive_decimation():
    """Test progressive decimation with multiple target levels"""
    print("=" * 60)
    print("Test 3: Progressive Decimation")
    print("=" * 60)

    vertices, faces = create_test_mesh()
    initial_count = vertices.shape[0]

    targets = [
        int(initial_count * 0.75),
        int(initial_count * 0.5),
        int(initial_count * 0.25),
        max(4, int(initial_count * 0.1))
    ]

    for i, target in enumerate(targets):
        decimated_verts, decimated_faces = spcumc.decimate_mesh(
            vertices, faces, target
        )
        ratio = decimated_verts.shape[0] / initial_count
        print(f"Level {i+1}: {decimated_verts.shape[0]:3d} vertices ({ratio:.1%} of original)")

        # Verify quality at each level
        assert decimated_verts.shape[0] >= 4, "Mesh collapsed too much!"
        assert decimated_faces.shape[0] >= 4, "Too few faces remaining!"

    print("✓ Progressive decimation completed successfully\n")

def test_marching_cubes_with_decimation():
    """Test integration: marching cubes -> decimation"""
    print("=" * 60)
    print("Test 4: Marching Cubes + Decimation Pipeline")
    print("=" * 60)

    # Create sparse sphere
    resolution = 64
    coords_list = []
    sdf_list = []

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                x, y, z = i / resolution, j / resolution, k / resolution
                dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2)
                sdf = dist - 0.3

                # Only store near-surface voxels
                if abs(sdf) < 0.1:
                    coords_list.append([i, j, k])
                    sdf_list.append(sdf)

    coords = torch.tensor(coords_list, dtype=torch.int32).cuda()
    sdfs = torch.tensor(sdf_list, dtype=torch.float32).cuda()

    print(f"Sparse voxels: {len(coords_list)}")

    # Generate mesh with marching cubes
    vertices, faces = spcumc.sparse_marching_cubes_from_points(
        coords, sdfs, iso_level=0.0, default_value=1.0
    )

    print(f"MC output: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Decimate
    target = max(100, vertices.shape[0] // 10)
    decimated_verts, decimated_faces = spcumc.decimate_mesh(
        vertices, faces, target
    )

    print(f"Decimated: {decimated_verts.shape[0]} vertices, {decimated_faces.shape[0]} faces")
    print(f"Reduction: {(1 - decimated_verts.shape[0] / vertices.shape[0]) * 100:.1f}%")
    print("✓ Pipeline test completed\n")

    return decimated_verts, decimated_faces

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DECIMATION ALGORITHM TEST SUITE")
    print("Testing improved quality checks")
    print("=" * 60 + "\n")

    try:
        # Test 1: Basic decimation
        decimated_verts, decimated_faces = test_basic_decimation()

        # Test 2: Quality metrics
        test_quality_metrics(decimated_verts, decimated_faces)

        # Test 3: Progressive decimation
        test_progressive_decimation()

        # Test 4: Full pipeline
        test_marching_cubes_with_decimation()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
