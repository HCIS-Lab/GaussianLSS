import torch
import json
import hydra
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm
from GaussianLSS.common import setup_config, setup_data_module

def setup(cfg):
    # Don't change these
    project_root = Path(hydra.utils.get_original_cwd())
    cfg.data.dataset_dir = project_root / cfg.data.dataset_dir
    cfg.data.labels_dir = project_root / cfg.data.labels_dir
    cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'

    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = False
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False
    cfg.loader.__delattr__("train_batch_size")
    cfg.loader.__delattr__("val_batch_size")

@hydra.main(config_path=str(Path.cwd() / 'config'), config_name='config.yaml')
def main(cfg):
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        {scene}.json
        {scene}/
            gt_box_{scene_token}.npz
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)

    data = setup_data_module(cfg)

    labels_dir = Path(cfg.data.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for split in ['val', 'train']:
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):
            scene_dir = labels_dir / episode.scene_name
            scene_dir.mkdir(exist_ok=True, parents=False)

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)
            info = []

            for i, batch in enumerate(tqdm(loader, position=1, leave=False)):
                info.extend(batch)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'
            scene_json.write_text(json.dumps(info))

if __name__ == '__main__':
    main()
