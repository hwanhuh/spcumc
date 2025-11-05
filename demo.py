import torch
import trimesh
import numpy as np

def create_sparse_sphere(voxel_res=128, radius=0.3, center_x=0, center_y=0, center_z=0, band=2/128):
    grid_values = np.linspace(0.0, 1.0, voxel_res)
    x, y, z = np.meshgrid(grid_values, grid_values, grid_values, indexing='ij')

    dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2)

    sdf_values = dist - radius
    mask = np.abs(sdf_values) <= band
    
    sparse_coords = np.stack(np.where(mask), axis=-1).astype(np.int32)
    sparse_sdfs = sdf_values[mask].astype(np.float32)

    coords_tensor = torch.from_numpy(sparse_coords)
    sdfs_tensor = torch.from_numpy(sparse_sdfs)

    return coords_tensor, sdfs_tensor

def save_obj_with_quads(filepath, verts, faces):
    verts_np = verts.cpu().numpy()
    faces_np = faces.cpu().numpy()

    with open(filepath, 'w') as f:
        for v in verts_np:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if faces_np.shape[-1] == 3:
            for face in faces_np:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        elif faces_np.shape[-1] == 4:
            for face in faces_np:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")

coords, feats = create_sparse_sphere()

vertices, faces = spcumc.sparse_dual_marching_cubes_from_points(
    coords.cuda(), 
    feats.float().cuda(), 
    0.0, 
)
save_obj_with_quads('output_sphere_mc.obj', vertices, faces)

vertices, faces = spcumc.sparse_marching_cubes_from_points(
    coords.cuda(), 
    feats.float().cuda(), 
    0.0, 
)
save_obj_with_quads('output_sphere_dmc.obj', vertices, faces)