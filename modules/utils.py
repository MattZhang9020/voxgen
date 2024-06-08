import random

import torch

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def get_voxel_map(voxel_coords, device='cpu', voxel_map_shape=(128, 128, 128)):
    voxel_map = torch.zeros(voxel_map_shape, dtype=torch.float, device=device)
    for x, y, z in voxel_coords:
        voxel_map[x, y, z] = 1.0
    return voxel_map


def plot_objt_by_dataset(dataset, target_idx, voxel_map_shape=(128, 128, 128), save=False):
    base_idx = sum(dataset.each_chair_part_counts[:target_idx])

    voxel_map = np.zeros(voxel_map_shape, dtype=np.float32)

    for i in range(base_idx, base_idx+dataset.each_chair_part_counts[target_idx]):
        for x, y, z in dataset.parts_voxel_coords[i]:
            voxel_map[x, y, z] = 1.0

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.voxels(np.moveaxis(voxel_map, 1, -1))

    if save:
        plt.savefig('fig.png')
    else:
        plt.show()


def plot_objt_by_models(encoder, decoder, dataset, target_idx, threshold, device='cpu', voxel_map_shape=(128, 128, 128), save=False):
    encoder.eval()
    decoder.eval()

    base_idx = sum(dataset.each_chair_part_counts[:target_idx])

    voxel_map = np.zeros(voxel_map_shape, dtype=np.float32)

    for i in range(base_idx, base_idx+dataset.each_chair_part_counts[target_idx]):
        voxel_coords = get_voxel_map(dataset[i], device).view(1, 1, *voxel_map_shape)
        latent = encoder(voxel_coords)
        pred = torch.sigmoid(decoder(latent))
        voxel_coords = (pred > threshold).nonzero()[:, 2:]

        for x, y, z in voxel_coords:
            voxel_map[x, y, z] = 1.0

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.voxels(np.moveaxis(voxel_map, 1, -1))

    if save:
        plt.savefig('fig.png')
    else:
        plt.show()


def plot_objt_by_latents(decoder, latents, threshold, voxel_map_shape=(128, 128, 128), save=False):
    decoder.eval()

    voxel_maps = []

    colors = [cm.rainbow(random.uniform(0, 1)) for _ in range(len(latents))]

    for latent in latents:
        voxel_map = np.zeros(voxel_map_shape, dtype=np.float32)
        pred = torch.sigmoid(decoder(latent.view(1, *latent.shape)))
        voxel_coords = (pred > threshold).nonzero()[:, 2:]

        for x, y, z in voxel_coords:
            voxel_map[x, y, z] = 1.0

        voxel_maps.append(voxel_map)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for voxel_map, color in zip(voxel_maps, colors):
        ax.voxels(np.moveaxis(voxel_map, 1, -1), facecolors=color)

    if save:
        plt.savefig('fig.png')
    else:
        plt.show()


def plot_part_by_voxel_coords(voxel_coords, save=False):
    voxel_map = get_voxel_map(voxel_coords).numpy()

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.voxels(np.moveaxis(voxel_map, 1, -1))

    if save:
        plt.savefig('fig.png')
    else:
        plt.show()


def dataloader_collate_fn(batch):
    return [torch.tensor(parts, dtype=torch.int) for parts in batch]
