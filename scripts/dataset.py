import os
import glob
import torch
from torch.utils.data import Dataset

class KeypointsSequenceDataset(Dataset):
    def __init__(self, data_paths, sequence_length, label_map, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.sequence_length = sequence_length

        for data_path in data_paths:
            for category in os.listdir(data_path):
                folder_path = os.path.join(data_path, category)
                label_name = category.split('_')[0]
                if label_name in label_map:
                    label = label_map[label_name]
                    keypoint_files = sorted(glob.glob(os.path.join(folder_path, 'keypoints_*.txt')))

                    if len(keypoint_files) > 0:
                        if len(keypoint_files) < sequence_length:
                            keypoint_files += [None] * (sequence_length - len(keypoint_files))
                        self.data.append((keypoint_files[:sequence_length], label))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoint_files, label = self.data[idx]

        sequence_data = []
        for keypoint_file in keypoint_files:
            if keypoint_file is None:
                keypoints = torch.zeros((17, 3), dtype=torch.float)  # Zero padding
            else:
                with open(keypoint_file, 'r') as f:
                    keypoints = eval(f.readlines()[0].split(', ', 1)[1])
                    keypoints = torch.tensor(keypoints, dtype=torch.float)
            sequence_data.append(keypoints)

        sequence_data = torch.stack(sequence_data)  # Shape: (sequence_length, 17, 3)
        return sequence_data, label
