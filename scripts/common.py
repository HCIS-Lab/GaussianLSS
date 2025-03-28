import hydra
from hydra import core, initialize, compose
from omegaconf import OmegaConf
import torch
from GaussianLSS.common import setup_experiment, load_backbone

def prepare_val(exp, device, CHECKPOINT_PATH=None, batch_size=1, mode='split'):
    core.global_hydra.GlobalHydra.instance().clear()    
    initialize(version_base="1.3", config_path='../config')
    cfg = compose(
        config_name='config',
        overrides=[          
            f'+experiment={exp}',
        ]
    )
    cfg.data.dataset_dir = f".{cfg.data.dataset_dir}"
    cfg.data.labels_dir = f".{cfg.data.labels_dir}"
    
    model, data, viz = setup_experiment(cfg)

    # load dataset
    if mode == 'split':
        SPLIT = 'val_qualitative_000'
        SUBSAMPLE = 5
        dataset = data.get_split(SPLIT, loader=False)
        dataset = torch.utils.data.ConcatDataset(dataset)
        dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else :
        dataset = data.get_split(mode, loader=False)
        dataset = torch.utils.data.ConcatDataset(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    print("Dataset length:",len(dataset))
    if CHECKPOINT_PATH is not None:
        network = load_backbone(CHECKPOINT_PATH, device=device, backbone=model.backbone)
        print("Loaded checkpoint.")
    else:
        network = model.backbone

    return model, network, loader, viz, dataset

def calculate_iou(model, network, loader, device, metric_mode):
    score_threshold = 0.6
    model.to(device)
    network.to(device)
    network.eval()
    with torch.no_grad():
        for i,batch in enumerate(loader):
    #         start = time.time()
            print(i,end='\r')
            for k, v in batch.items():
                if k!='features' or k!='center':
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    elif isinstance(v, list):
                        if isinstance(v[0],torch.Tensor):
                            batch[k] = [_v.to(device) for _v in v]
                    else:
                        batch[k] = v

            pred = network(batch)
            model.metrics.update(pred,batch)
            # break

    # model.metrics.compute()
    # print()
    print(model.metrics.compute())