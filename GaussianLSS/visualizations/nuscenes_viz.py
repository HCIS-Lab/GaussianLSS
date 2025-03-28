import torch
import numpy as np
import cv2
from matplotlib.pyplot import get_cmap

# many colors from
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = {
    # static
    'lane':                 (110, 110, 110),
    'road_segment':         (90, 90, 90),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (255, 200, 0),

    # dynamic
    'vehicle':              (255, 158, 0),
    'ped':                  (0, 0, 230),

    # other toplogy
    'drivable_area':        (255, 127, 80),
    'ped_crossing':         (255, 61, 99),
    'walkway':              (0, 207, 191),
    'carpark_area':         (34, 139, 34),
    'stop_line':            (138, 43, 226),

    'nothing':              (200, 200, 200)
}

def colorize(x, colormap='winter'):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f'COLORMAP_{colormap.upper()}'))

def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)


def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)


class NuScenesViz:
    def __init__(self, keys=[], bev_h=200, bev_w=200):
        self.keys = keys
        self.bev_h = bev_h
        self.bev_w = bev_w
    
    def draw_ego(self, img, view):

        points = np.array([
                [-4.0 / 2 + 0.3, -1.73 / 2, 1],
                [-4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3, -1.73 / 2, 1],
            ])
        
        points = view @ points.T
        cv2.fillPoly(img, [points.astype(np.int32)[:2].T], color=(164, 0, 0))
        return img

    def draw_image(self, image, tensor, color):
        mask = tensor.cpu().numpy().astype(bool)
        image[mask] = color
        return image
    
    # We plot the BEV in this order:
    # map_keys -> dynamic_keys 
    def visualize_gt(self, batch, view, index):
        image = np.zeros((self.bev_h, self.bev_w, 3)).astype(np.uint8)
        image += 200
        for name in self.keys:
            image = self.draw_image(image, batch[name][index, 0], COLORS[name])
            
        image = self.draw_ego(image, view)
        
        return [image]
    
    def visualize_pred(self, pred, view, index, batch):
        image = np.zeros((self.bev_h, self.bev_w, 3)).astype(np.uint8)
        image += 200
        for name in self.keys:
            try:
                prediction = (pred[name][index, 0].detach().sigmoid() > 0.4)
            except:
                prediction = batch[name][index, 0]
            image = self.draw_image(image, prediction, COLORS[name])

        image = self.draw_ego(image, view)
        
        return [image]

    @torch.no_grad()
    def visualize(self, batch, pred, b_max=8, **kwargs):
        view = batch['view'][0].cpu().numpy()
        batch_size = batch['view'].shape[0]
        for b in range(min(batch_size, b_max)):
            gt_viz = self.visualize_gt(batch, view, b)
            pred_viz = self.visualize_pred(pred, view, b, batch)
            right = gt_viz + pred_viz
            
            for x in right:
                x[:,-1] = [0,0,0]
                
            right = [x for x in right if x is not None]
            right = np.hstack(right)

            image = None if not hasattr(batch.get('image'), 'shape') else batch['image']

            if image is not None:
                imgs = [to_image(image[b][i]) for i in range(image.shape[1])]

                if len(imgs) == 6:
                    a = np.hstack(imgs[:3])
                    b = np.hstack(imgs[3:])
                    left = resize(np.vstack((a, b)), right)
                else:
                    left = np.hstack([resize(x, right) for x in imgs])

                yield np.hstack((left, right))
            else:
                yield right

    def __call__(self, batch=None, pred=None, **kwargs):
        return list(self.visualize(batch=batch, pred=pred, **kwargs))
