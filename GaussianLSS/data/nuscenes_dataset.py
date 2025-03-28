import torch
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

from .common import get_pose, get_split, get_view_matrix
from .transforms import Sample, SaveDataTransform

DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
    'emergency',
]

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    **dataset_kwargs
):
    helper = NuScenesSingleton(dataset_dir, version)
    transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    result = list()

    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue
        data = NuScenesDataset(scene_name, scene_record, helper,
                               transform=transform, **dataset_kwargs)
        result.append(data)

    return result


class NuScenesSingleton:
    """
    Wraps nuScenes API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    RADARS = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        helper: NuScenesSingleton,
        bev: dict,
        cameras=[[0, 1, 2, 3, 4, 5]],
        transform=None,
        **kwargs
    ):
        self.view = get_view_matrix(**bev)
        self.scene_name = scene_name
        self.transform = transform
        self.nusc = helper.nusc
        self.map_name = self.nusc.get('log', scene_record['log_token'])['location']
        self.samples = self.parse_scene(scene_record, cameras)

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample_record(self, sample_record, camera_rig):
        """
            box: world coordinate
            parse_pose: pose @ point , LiDAR -> world 
                   inv: pose_inv @ point , LiDAR <- world
            sensor, ego_sensor world 
        """
        
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])

        # calibrated_lidar = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

        # cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat # @ world_from_lidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            # cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

        return {
            'scene': self.scene_name,
            'map_name': self.map_name,
            'token': sample_record['token'],
            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),
            'lidar_record': egolidar,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
        }

    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            # if int(a['visibility_token']) == 1:
            #     continue
            if idx is not None:
                result[idx].append(a)

        return result
    
    def get_gt_box(self, lidar_record, anns_by_category):
        from nuscenes.utils import data_classes
        """ 
            Return: 
                List[bounding boxes] 
                bounding boxes 8 dimensions: cx,cy,w,l,cz,h,yaw,class
        """
        gt_boxes = []
        for class_index, anns in enumerate(anns_by_category):
            for ann in anns:
                box = data_classes.Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

                # project box global -> lidar
                yaw = Quaternion(lidar_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(lidar_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                cx, cy, cz = box.center
                w, l, h = box.wlh
                yaw = box.orientation.yaw_pitch_roll[0]
                gt_boxes.append(np.array([cx, cy, l, w, cz, h, yaw, class_index, int(ann['visibility_token'])]))
        
        gt_boxes = np.stack(gt_boxes, 0) if len(gt_boxes) != 0 else np.zeros((0,9))
        return gt_boxes
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Raw annotations
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC)
        gt_box = self.get_gt_box(sample['lidar_record'], anns_dynamic)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            gt_box=gt_box,
            **sample
        )
        data = self.transform(data)
        return data
