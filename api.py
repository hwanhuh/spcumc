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
