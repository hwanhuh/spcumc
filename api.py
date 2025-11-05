import torch
import spcumc._backend as _backend

def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):
    """
    Performs sparse marching cubes on a set of voxels.

    Args:
        coords (torch.Tensor): Integer tensor of shape [N, 3] with the (x, y, z) coordinates of the voxels.
        corners (torch.Tensor): Float tensor of shape [N, 8] with the scalar values at the 8 corners of each voxel.
        iso (float): The iso-level to contour at.
        ensure_consistency (bool): If True, averages corner values at shared locations to ensure a consistent mesh.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - verts: Float tensor of shape [V, 3] with the vertex positions.
            - tris: Int tensor of shape [T, 3] with the triangle indices.
    """
    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()
    
    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    verts, tris = _backend.sparse_marching_cubes(coords, corners, iso, ensure_consistency)

    return verts, tris

def get_cube_indices(corners, iso):
    """
    Calculates the marching cubes case index for each voxel based on corner values.

    Args:
        corners (torch.Tensor): Float tensor of shape [N, 8] with the scalar values at the 8 corners of each voxel.
        iso (float): The iso-level to contour at.

    Returns:
        torch.Tensor: An integer tensor of shape [N] with the case index for each voxel.
    """
    if not corners.is_cuda: corners = corners.cuda()
    
    v = corners.float()
    cube_indices = (v < iso).int()
    cube_indices = (
        cube_indices[:, 0] * 1 |
        cube_indices[:, 1] * 2 |
        cube_indices[:, 2] * 4 |
        cube_indices[:, 3] * 8 |
        cube_indices[:, 4] * 16 |
        cube_indices[:, 5] * 32 |
        cube_indices[:, 6] * 64 |
        cube_indices[:, 7] * 128
    ).int()
    
    return cube_indices

def sparse_marching_cubes_from_points(coords, point_values, iso, default_value=1.0):
    """
    Performs sparse marching cubes from a sparse grid of scalar values.

    This function uses a high-performance CUDA backend to construct the 8 corner
    values for each active voxel by looking up values in the provided sparse grid.

    Args:
        coords (torch.Tensor): Integer tensor of shape [N, 3] with the (x, y, z)
            coordinates of the active voxels. These are also the points in the grid
            with defined scalar values.
        point_values (torch.Tensor): Float tensor of shape [N] or [N, 1] with the
            scalar value at each coordinate in `coords`.
        iso (float): The iso-level to contour at.
        default_value (float): The scalar value to assume for corner points that are
            not in the input `coords`. This should be a value that is known to be
            either completely inside or completely outside the surface.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - verts: Float tensor of shape [V, 3] with the vertex positions.
            - tris: Int tensor of shape [T, 3] with the triangle indices.
    """
    if not coords.is_cuda: coords = coords.cuda()
    if not point_values.is_cuda: point_values = point_values.cuda()

    coords = coords.int().contiguous()
    point_values = point_values.float().contiguous().flatten()

    verts, tris = _backend.sparse_marching_cubes_from_points(coords, point_values, iso, default_value)

    return verts, tris

def sparse_dual_marching_cubes(coords, corners, iso):
    """
    Performs sparse dual marching cubes on a set of voxels.

    Dual marching cubes generates quad-based meshes instead of triangle meshes.
    Each active voxel generates a dual vertex at its center, and quads are formed
    by connecting dual vertices across shared marching cube edges.

    Args:
        coords (torch.Tensor): Integer tensor of shape [N, 3] with the (x, y, z) coordinates of the voxels.
        corners (torch.Tensor): Float tensor of shape [N, 8] with the scalar values at the 8 corners of each voxel.
        iso (float): The iso-level to contour at.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - verts: Float tensor of shape [V, 3] with the vertex positions (dual vertices).
            - quads: Int tensor of shape [Q, 4] with the quad indices.
    """
    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()

    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    verts, quads = _backend.sparse_dual_marching_cubes(coords, corners, iso)

    return verts, quads

def sparse_dual_marching_cubes_from_points(coords, point_values, iso, default_value=1.0):
    """
    Performs sparse dual marching cubes from a sparse grid of scalar values.

    This function builds corner values from a sparse grid and then generates
    a quad-based mesh using the dual marching cubes algorithm.

    Args:
        coords (torch.Tensor): Integer tensor of shape [N, 3] with the (x, y, z)
            coordinates of the active voxels.
        point_values (torch.Tensor): Float tensor of shape [N] or [N, 1] with the
            scalar value at each coordinate in `coords`.
        iso (float): The iso-level to contour at.
        default_value (float): The scalar value to assume for corner points that are
            not in the input `coords`.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - verts: Float tensor of shape [V, 3] with the vertex positions (dual vertices).
            - quads: Int tensor of shape [Q, 4] with the quad indices.
    """
    if not coords.is_cuda: coords = coords.cuda()
    if not point_values.is_cuda: point_values = point_values.cuda()

    coords = coords.int().contiguous()
    point_values = point_values.float().contiguous().flatten()

    verts, quads = _backend.sparse_dual_marching_cubes_from_points(coords, point_values, iso, default_value)

    return verts, quads

def decimate_mesh(vertices, faces, target_vertex_count):
    """
    Decimate a triangle mesh using quadric error metrics (QEM).

    This function reduces the number of vertices in a mesh by iteratively
    collapsing edges, while minimizing geometric error using quadric error metrics.

    Args:
        vertices (torch.Tensor): Float tensor of shape [V, 3] with vertex positions.
        faces (torch.Tensor): Integer tensor of shape [F, 3] with triangle indices.
        target_vertex_count (int): Target number of vertices after decimation.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - vertices: Float tensor of shape [V', 3] with decimated vertex positions.
            - faces: Integer tensor of shape [F', 3] with decimated triangle indices.

    Note:
        The actual number of vertices in the output may be slightly different
        from target_vertex_count depending on the mesh topology.
    """
    vertices = vertices.float().contiguous()
    faces = faces.int().contiguous()

    new_vertices, new_faces = _backend.decimate_mesh(vertices, faces, target_vertex_count)

    return new_vertices, new_faces
