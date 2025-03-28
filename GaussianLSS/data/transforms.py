import pathlib

import torch
import torchvision
import numpy as np
import cv2

from PIL import Image
from .common import INTERPOLATION, sincos2quaternion
from .augmentations import RandomTransformImage, RandomTransformationBev
from nuscenes.utils.data_classes import Box

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

nusc_map_name = [
    'boston-seaport',
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown'
]

map_layer = [
    'lane', 'road_segment',
    'road_divider', 'lane_divider',
    'ped_crossing','walkway','carpark_area',
]

class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        map_name,
        intrinsics,
        extrinsics,
        images,
        view,
        gt_box,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene
        self.view = view
        self.map_name = map_name
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.gt_box = gt_box

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics
        }

    def get_box(self, batch: Sample):
        scene_dir = self.labels_dir / batch.scene
        gt_box_path = f'gt_box_{batch.token}.npz'
        gt_box = batch.gt_box
        np.savez_compressed(scene_dir / gt_box_path, gt_box=gt_box)
        return {'gt_box': gt_box_path}

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_box(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, 
                dataset_dir, 
                labels_dir, 
                image_config, 
                image_data=True, 
                box='',
                split_intrin_extrin=False, 
                augment_img=False, 
                augment_bev=False, 
                img_params=None, 
                bev_aug_conf=None, 
                training=True, 
                bev=True,
                vehicle=False,
                ped=False,
                map_layers=[],
                **kwargs
        ):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.image_data = image_data
        self.bev = bev
        self.box = box
        self.split_intrin_extrin = split_intrin_extrin
        self.img_transform = torchvision.transforms.ToTensor()

        self.vehicle = vehicle
        self.ped = ped
        self.map_layers = map_layers

        self.training = training
        self.augment_img = RandomTransformImage(img_params, training) if augment_img else None
        self.augment_bev = RandomTransformationBev(bev_aug_conf, training) if augment_bev else None

        self.to_tensor = super().__call__

        if len(self.map_layers) > 0:
            self._init_map()
    
    def _init_map(self):
        from nuscenes.map_expansion.map_api import NuScenesMap

        print("Initializing nuScenes map...")
        self.nusc_map = {}
        for map_name in nusc_map_name:
            self.nusc_map[map_name] = NuScenesMap(dataroot=self.dataset_dir, map_name=map_name)

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()
        lidar2img = list()

        for image_path, I_original, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
            images.append(self.to_tensor(image_new))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop
            extrinsic = np.float32(extrinsic)
            
            if not self.split_intrin_extrin:
                viewpad = np.float32(np.eye(4))
                viewpad[:I.shape[0], :I.shape[1]] = I
                lidar2img.append(torch.tensor(viewpad @ extrinsic))
            else:
                intrinsics.append(torch.tensor(I))

        result = {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
        }

        sensor = {}
        if not self.split_intrin_extrin:
            sensor = {
                'lidar2img': torch.stack(lidar2img, 0),
            }
        else:
            sensor = {
                'intrinsics': torch.stack(intrinsics, 0),
                'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
            }

        result.update(sensor)    
        return result
    
    def get_cameras_augm(self, sample: Sample, **kwargs):
        images = list()
        intrinsics = list()
        extrinsics = list()

        for image_path, intrinsic, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):
            image = Image.open(self.dataset_dir / image_path)
            images.append(image)

            intrinsic = np.float32(intrinsic)
            extrinsic = np.float32(extrinsic)

            # intrinsic = add_intrinsic_noise(intrinsic)
            # extrinsic = add_extrinsic_noise(extrinsic)

            intrinsics.append(intrinsic)
            extrinsics.append(torch.tensor(extrinsic))

        result = {'image': images}
        result.update({'intrinsics':intrinsics, 'extrinsics':extrinsics})
        result = self.augment_img(result)
        result['image'] = torch.stack(result['image'], 0)

        result['intrinsics'] = torch.stack(result['intrinsics'], 0)
        result['extrinsics'] = torch.stack(result['extrinsics'], 0)

        lidar2img = list()
        for intrinsic,  extrinsic in zip(result['intrinsics'], result['extrinsics']):
            viewpad = torch.eye(4, dtype=torch.float32)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img.append(viewpad @ extrinsic)
        result['lidar2img'] = torch.stack(lidar2img, 0)

        # result['cam_idx'] = torch.LongTensor(sample.cam_ids)

        return result

    # copied from PointBEV
    def _prepare_augmented_boxes(self, bev_aug, points, inverse=True):
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))
        if inverse:
            # Inverse query aug:
            # Ex: when tx=10, the query is 10/res meters front,
            # so points are fictivelly 10/res meters back.
            Rquery[:3, :3] = bev_aug[:3, :3].T
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ (X + tquery)
            points_out = (Rquery @ (points_in[:3, :] + tquery))
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = ((Rquery @ points_in[:3, :]) + tquery)

        return points_out    
    
    def get_bev_from_gtbbox(self, sample: Sample, bev_augm, mode='vehicle'):

        scene_dir = self.labels_dir / sample.scene
        gt_box = np.load(scene_dir / sample.gt_box, allow_pickle=True)['gt_box']
        V = sample.view

        # bev segmentation
        bev = np.zeros((200, 200), dtype=np.uint8)

        # center & offset
        center_score = np.zeros((200, 200), dtype=np.float32)
        center_offset = np.zeros((200, 200, 2), dtype=np.float32) #+ 255
        visibility = np.full((200, 200), 255, dtype=np.uint8)

        buf = np.zeros((200, 200), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(200), np.arange(200)), -1).astype(np.float32)
        sigma = 1

        for box_data in gt_box:
            if len(box_data) == 0:
                continue
            class_idx = int(box_data[7])
            if class_idx == 5 and mode == 'vehicle': 
                continue
            elif class_idx != 5 and mode == 'ped':
                continue
            translation = [box_data[0],box_data[1],box_data[4]]
            size = [box_data[2],box_data[3],box_data[5]]
            yaw = box_data[6]
            yaw = -yaw - np.pi / 2
            visibility_token = box_data[8]
            box = Box(translation, size, sincos2quaternion(np.sin(yaw),np.cos(yaw)))
            points = box.bottom_corners()

            center = points.mean(-1)[:, None] # unsqueeze 1

            homog_points = np.ones((4, 4))
            homog_points[:3, :] = points
            homog_points[-1, :] = 1
            points = self._prepare_augmented_boxes(bev_augm, homog_points)
            points[2] = 1 # add 1 for next matrix matmul
            points = (V @ points)[:2]
            cv2.fillPoly(bev, [points.round().astype(np.int32).T], 1, INTERPOLATION)

            # center, offsets, height
            homog_points = np.ones((4, 1))
            homog_points[:3, :] = center
            homog_points[-1, :] = 1
            center = self._prepare_augmented_boxes(bev_augm, homog_points).astype(np.float32)
            center[2] = 1 # add 1 for next matrix matmul
            center = (V @ center)[:2, 0].astype(np.float32) # squeeze 1

            buf.fill(0)
            cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0
            # center_offset[mask] = center[None] - coords[mask]
            center_off = center[None] - coords
            center_offset[mask] = center_off[mask]
            g = np.exp(-(center_off ** 2).sum(-1) / (2 * sigma ** 2))
            center_score = np.maximum(center_score, g)
            # center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (2 * sigma ** 2))
            
            # visibility
            visibility[mask] = visibility_token
        
        bev = self.to_tensor(255 * bev)
        center_score = self.to_tensor(center_score)
        center_offset = self.to_tensor(center_offset)
        visibility = torch.from_numpy(visibility)
        
        return bev, center_score, center_offset, visibility
    
    def get_map(self, sample: Sample, bev_augm):
        h, w = 200, 200
        V = np.array(sample.view)
        pose = sample['pose'] @ bev_augm
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        lidar2global = (V @ S @ np.linalg.inv(pose))
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        angle = (yaw / np.pi * 180)

        pose = sample['pose'] @ bev_augm

        map_mask = self.nusc_map[sample['map_name']].get_map_mask((pose[0][-1], pose[1][-1], 100, 100), angle, self.map_layers, (h,w))
        result = {self.map_layers[i]: self.to_tensor(255 * np.flipud(m)[..., None]) for i, m in enumerate(map_mask)}
        return result
    
    
    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        
        result = dict()
        result['view'] = torch.tensor(batch.view)
        result['token'] = batch['token']
        result['map_name'] = batch['map_name']
        result['pose'] = np.float32(batch['pose'])
        result['pose_inverse'] = np.float32(batch['pose_inverse'])

        if self.image_data:
            get_cameras = self.get_cameras_augm if self.augment_img is not None else self.get_cameras 
            result.update(get_cameras(batch, **self.image_config))

        if self.bev:
            bev_augm = self.augment_bev() if self.augment_bev else np.eye(4) 
            
            if self.vehicle:
                bev, center, offset, visibility = self.get_bev_from_gtbbox(batch, bev_augm, mode='vehicle')
                result['vehicle'] = bev
                result['vehicle_center'] = center
                result['vehicle_offset'] = offset
                result['vehicle_visibility'] = visibility
            
            if self.ped:
                bev, center, offset, visibility = self.get_bev_from_gtbbox(batch, bev_augm, mode='ped')
                result['ped'] = bev
                result['ped_center'] = center
                result['ped_offset'] = offset
                result['ped_visibility'] = visibility

            if len(self.map_layers) > 0:
                gt_map = self.get_map(batch, bev_augm)
                result.update(gt_map)
                
            bev_augm = torch.from_numpy(bev_augm)
            result['extrinsics'] = result['extrinsics'] @ bev_augm
            result['lidar2img'] = result['lidar2img'] @ bev_augm
            result['bev_augm'] = bev_augm

        return result