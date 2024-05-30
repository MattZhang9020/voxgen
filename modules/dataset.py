import os
import re

import numpy as np

from concurrent import futures

from .utils import rotate_objt_along_axis

from torch.utils.data import Dataset
from tqdm import tqdm


def sorted_char(list):
    return sorted(list, key=lambda key: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', key)])


class VoxelDataset(Dataset):
    def __init__(self,
                 data_dir_pth,
                 each_chair_part_counts_npy_pth,
                 outlier_objt_indices_npy_pth=None,
                 designate_num_objts=None,
                 train_test_split_ratio_train=0.8,
                 is_train=True):

        self.data_dir_pth = data_dir_pth
        self.each_chair_part_counts_npy_pth = each_chair_part_counts_npy_pth
        self.outlier_objt_indices_npy_pth = outlier_objt_indices_npy_pth

        self.outlier_objt_indices = np.load(self.outlier_objt_indices_npy_pth) if self.outlier_objt_indices_npy_pth != None else None

        self.each_chair_part_counts = np.load(self.each_chair_part_counts_npy_pth)[:designate_num_objts]
        self.num_objts = len(self.each_chair_part_counts)
        
        self._train_test_split(train_test_split_ratio_train, is_train)

        if type(self.outlier_objt_indices) != None:
            self.data_file_names = self._get_data_names_excluding_outliers()
        else:
            self.data_file_names = np.array(sorted_char(os.listdir(self.data_dir_pth)), dtype=str)
        self.num_parts = len(self.data_file_names)

        self.parts_voxel_coords = self._load_voxel_data()
        
    def _train_test_split(self, train_test_split_ratio_train, is_train):
        if is_train:
            self.num_objts = int(len(self.each_chair_part_counts) * train_test_split_ratio_train)
            self.each_chair_part_counts = self.each_chair_part_counts[:self.num_objts]
        else:
            num_train = int(len(self.each_chair_part_counts) * train_test_split_ratio_train)
            self.num_objts = len(self.each_chair_part_counts) - num_train
            self.each_chair_part_counts = self.each_chair_part_counts[num_train : num_train + self.num_objts]

    def _get_data_names_excluding_outliers(self):
        data_file_names = np.array(sorted_char(os.listdir(self.data_dir_pth)), dtype=str)
        data_file_names_no_outlier = []

        base_index = 0

        outlier_objt_indices_set = set(self.outlier_objt_indices)

        for i, parts_count in enumerate(self.each_chair_part_counts):
            if i not in outlier_objt_indices_set:
                data_file_names_no_outlier.extend(data_file_names[base_index:base_index+parts_count])
            base_index += parts_count

        self.each_chair_part_counts = [count for i, count in enumerate(self.each_chair_part_counts) if i not in self.outlier_objt_indices]
        self.num_objts = len(self.each_chair_part_counts)

        return data_file_names_no_outlier

    def _load_voxel_data(self):
        print('Trying to load {} objects with total {} parts into memory.'.format(self.num_objts, self.num_parts))

        def load_data(data_pth):
            return rotate_objt_along_axis(np.load(data_pth), rotation_angle=180, axis='y')

        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            data_file_paths = [os.path.join(self.data_dir_pth, file_name) for file_name in self.data_file_names]
            parts_voxel_coords = list(tqdm(executor.map(load_data, data_file_paths), total=len(data_file_paths)))

        return parts_voxel_coords

    def __len__(self):
        return self.num_parts

    def __getitem__(self, idx):
        return self.parts_voxel_coords[idx]
