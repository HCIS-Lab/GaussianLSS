import json
import torch

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    image=None,                         # image config
    **dataset_kwargs
):
    out = []
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    training = True if split == 'train' else False
    transform = LoadDataTransform(dataset_dir, labels_dir, image, training=training, **dataset_kwargs)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    for s in split_scenes:
        tmp_dataset = NuScenesGeneratedDataset(s, labels_dir, transform=transform)
        out.append(tmp_dataset)
    return out

class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx=None, token=None):
        if token is not None:
            for i, sample in enumerate(self.samples):
                if token == sample['token']:
                    idx = i 
        data = Sample(**self.samples[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data