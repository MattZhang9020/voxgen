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
                 train_test_split_ratio_train=0.9,
                 is_train=True):

        self.data_dir_pth = data_dir_pth
        self.each_chair_part_counts_npy_pth = each_chair_part_counts_npy_pth
        self.outlier_objt_indices_npy_pth = outlier_objt_indices_npy_pth
        self.train_test_split_ratio_train = train_test_split_ratio_train
        self.is_train = is_train

        self.each_chair_part_counts = np.load(self.each_chair_part_counts_npy_pth)[:designate_num_objts]
        self.num_objts = len(self.each_chair_part_counts)
        
        total_objts = self.num_objts
        self.num_train_objts = int(total_objts * self.train_test_split_ratio_train)
        self.num_train_parts = sum(self.each_chair_part_counts[:self.num_train_objts])
        
        self.data_file_names = np.array(sorted_char(os.listdir(self.data_dir_pth)), dtype=str)
        self.num_parts = len(self.data_file_names)
        
        self._train_test_split()
        
        if outlier_objt_indices_npy_pth != None:
            self._excluding_outliers()

        self.parts_voxel_coords = self._load_voxel_data()
        
    def _train_test_split(self):
        if self.is_train:
            self.each_chair_part_counts = self.each_chair_part_counts[:self.num_train_objts]
            self.num_objts = len(self.each_chair_part_counts)
            
            self.data_file_names = self.data_file_names[:self.num_train_parts]
            self.num_parts = len(self.data_file_names)
        else:            
            self.each_chair_part_counts = self.each_chair_part_counts[self.num_train_objts:]
            self.num_objts = len(self.each_chair_part_counts)
            
            self.data_file_names = self.data_file_names[self.num_train_parts:]
            self.num_parts = len(self.data_file_names)

    def _excluding_outliers(self):
        each_chair_part_counts_no_outlier = []
        data_file_names_no_outlier = []

        outlier_objt_indices = np.load(self.outlier_objt_indices_npy_pth)
        
        # making set is for acceleration purpose
        outlier_objt_indices_set = set(outlier_objt_indices)
        
        base_index = 0
        for i, parts_count in enumerate(self.each_chair_part_counts):
            if not self.is_train:
                i += self.num_train_parts
                
            if i not in outlier_objt_indices_set:
                each_chair_part_counts_no_outlier.append(parts_count)
                data_file_names_no_outlier.extend(self.data_file_names[base_index:base_index+parts_count])
            
            base_index += parts_count

        self.each_chair_part_counts = each_chair_part_counts_no_outlier
        self.num_objts = len(self.each_chair_part_counts)
        
        self.data_file_names = data_file_names_no_outlier
        self.num_parts = len(self.data_file_names)

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
