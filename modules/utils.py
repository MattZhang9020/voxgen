import torch

import numpy as np
import matplotlib.pyplot as plt


def rotate_objt_along_axis(voxel_coords, rotation_angle, axis, voxel_map_shape=(128, 128, 128)):
    theta = np.radians(rotation_angle)

    if axis == 'x':
        rot_matrix = np.array([[1, 0, 0],
                               [0, np.cos(theta), -np.sin(theta)],
                               [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                               [0, 1, 0],
                               [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    center = np.array(voxel_map_shape) / 2
    centered_coordinates = voxel_coords - center

    rotated_coords = np.dot(centered_coordinates, rot_matrix.T)
    rotated_coords += center

    rotated_coords = np.round(rotated_coords).astype(int)
    valid_indices = np.all((rotated_coords >= 0) & (rotated_coords < np.array(voxel_map_shape)), axis=1)

    rotated_coords = rotated_coords[valid_indices]

    return rotated_coords


def plot_objt(dataset, target_idx, voxel_map_shape=(128, 128, 128)):
    base_idx = sum(dataset.each_chair_part_counts[:target_idx])

    voxel_map = np.zeros(voxel_map_shape, dtype=np.float32)

    for i in range(base_idx, base_idx+dataset.each_chair_part_counts[target_idx]):
        for x, y, z in dataset.voxel_coords[i]:
            voxel_map[x, y, z] = 1.0

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.voxels(np.moveaxis(voxel_map, 1, -1))

    plt.show()
    

def dataloader_collate_fn(batch):
    return torch.nested.nested_tensor(batch, dtype=torch.int)


def get_occurrence_map(voxel_coords, voxel_map_shape=(128, 128, 128)):
    occurrence_map = torch.zeros(voxel_map_shape, dtype=torch.float32)
    for coords in voxel_coords:
        occurrence_map[coords] = 1.0
    return occurrence_map
