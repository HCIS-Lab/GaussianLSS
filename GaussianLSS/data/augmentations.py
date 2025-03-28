from scipy.spatial.transform import Rotation as R

import torchvision
import torch
import numpy as np

from PIL.ImageTransform import AffineTransform

class RandomTransformImage(object):
    def __init__(self, img_params=None, training=True):
        self.img_params = img_params
        self.training = training
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, results):
        final_dims = self.img_params["final_dim"][::-1]
        for i in range(len(results['image'])):
            # img = Image.fromarray(np.uint8(results['img'][i]))
            img = results['image'][i]
            scale, resize_dims, crop, flip, rotate, crop_zoom, zoom = self.sample_augmentation()

            ida_mat = self.get_affinity_matrix_from_augm(
                scale, crop[1], crop_zoom, flip, rotate, final_dims, img.size
            )
            img = self.pil_preprocess_from_affine_mat(img, ida_mat, final_dims)
            if 'depth' in results:
                depth = results['depth'][i]
                depth = self.pil_preprocess_from_affine_mat(depth, ida_mat, final_dims)
                results['depth'][i] = self.transform(depth)
                
            results['image'][i] = self.transform(img)
            results['intrinsics'][i] = torch.tensor(ida_mat @ results['intrinsics'][i])
            results['ida_mat'] = torch.tensor(ida_mat)

        return results

    def get_affinity_matrix_from_augm(
        self, scale, crop_sky, crop_zoom, flip, rotate, final_dims, W_H=(1600, 900)
    ):        
        res = list(W_H)

        affine_mat = np.eye(3)
        # Resize scaling factor.
        affine_mat[:2, :2] *= scale
        # Update res.
        res = [_ * scale for _ in res]

        # Centered crop zoom.
        w, h = final_dims
        affine_mat[0, :2] *= w / (crop_zoom[2] - crop_zoom[0])
        affine_mat[1, :2] *= h / (crop_zoom[3] - crop_zoom[1])
        affine_mat[0, 2] += (w - res[0] * w / (crop_zoom[2] - crop_zoom[0])) / 2
        affine_mat[1, 2] += (
            h - (res[1] + crop_sky) * h / (crop_zoom[3] - crop_zoom[1])
        ) / 2

        # Flip
        if flip:
            flip_mat = np.eye(3)
            flip_mat[0, 0] = -1
            flip_mat[0, 2] += w
            affine_mat = flip_mat @ affine_mat

        # Rotate
        theta = -rotate * np.pi / 180
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x, y = w / 2, h / 2
        rot_center_mat = np.array(
            [
                [cos_theta, -sin_theta, -x * cos_theta + y * sin_theta + x],
                [sin_theta, cos_theta, -x * sin_theta - y * cos_theta + y],
                [0, 0, 1],
            ]
        )
        affine_mat = rot_center_mat @ affine_mat
        return affine_mat.astype(np.float32)
    
    def pil_preprocess_from_affine_mat(self, img, affine_mat, final_dims):
        inv_mat = np.linalg.inv(affine_mat)
        img = img.transform(
            size=tuple(final_dims), method=AffineTransform(inv_mat[:2].ravel())
        )
        return img

    def sample_augmentation(self):
        """Corresponds to get_resizing_and_cropping_parameters in the original code with some improvements.
        Available transformations:
            - scale
            - crop sky
            - crop zoom
            - final scale
            - flip
            - rotate.

        Ex: [1600,900] -> scale: 0.5 [800,450] -> crop sky: 10 [800,440] -> ...
        """
        # Specify the input image dimensions
        H, W = self.img_params["H"], self.img_params["W"]

        # During training
        if self.training:
            # Randomly choose a resize factor, e.g: 0.3.
            scale = np.random.uniform(*self.img_params["scale"])

            # Resize images, e.g: [270,480]
            newW, newH = int(W * scale), int(H * scale)

            # Resize.
            resize_dims = (newW, newH)

            # Crop the sky.
            crop_h = int(
                (1 - np.random.uniform(*self.img_params["crop_up_pct"])) * newH
            )
            crop = (0, crop_h, newW, newH)

            # Zoom in, zoom out: neutral=1, e.g: [0.95,1.05]
            zoom = np.random.uniform(*self.img_params["zoom_lim"])
            crop_zoomh, crop_zoomw = (
                ((newH - crop_h) * (1 - zoom)) // 2,
                (newW * (1 - zoom)) // 2,
            )
            crop_zoom = (
                -crop_zoomw,
                -crop_zoomh,
                crop_zoomw + newW,
                crop_zoomh + newH - crop_h,
            )

            # Allow flip and rotate during training.
            flip = False
            if self.img_params["rand_flip"] and np.random.choice([0, 1]):  # False
                flip = True
            rotate = np.random.uniform(*self.img_params["rot_lim"])  # ~U(0,0)
        else:
            # Randomly choose a resize factor, e.g: 0.3.
            # Images: [900,1600]
            scale = np.mean(self.img_params["scale"])

            # Resize images, e.g: [270,480]
            newW, newH = int(W * scale), int(H * scale)  # 480, 270

            # Resize.
            resize_dims = (newW, newH)

            # Remove the sky.
            crop_h = int((1 - np.mean(self.img_params["crop_up_pct"])) * newH)
            crop = (0, crop_h, newW, newH)

            # Zoom inside image.
            zoom = 1.0
            crop_zoom = (0, 0, newW, newH - crop_h)

            # Flip and rotate
            flip = False
            rotate = 0

        return scale, resize_dims, crop, flip, rotate, crop_zoom, zoom
    
class RandomTransformationBev(object):
    def __init__(self, bev_aug_conf=None, training=True):
        self.training = training
        self.bev_aug_conf = bev_aug_conf

    def get_random_ref_matrix(self):
        """
        Use scipy to create a random reference transformation matrix.
        """
        coeffs = self.bev_aug_conf
        trans_coeff, rot_coeff = coeffs[:3], coeffs[3:]

        # Initialize in homogeneous coordinates.
        mat = np.eye(4, dtype=np.float32)

        # Translate
        mat[:3, 3] = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
            trans_coeff
        )

        # Rotate
        random_zyx = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
            rot_coeff
        )
        mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

        return mat
    
    def __call__(self):
        if self.training:
            return self.get_random_ref_matrix()
        else:
            return np.eye(4, dtype=np.float32)
    