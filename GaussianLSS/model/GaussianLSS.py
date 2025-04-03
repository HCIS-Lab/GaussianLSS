import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class GaussianLSS(nn.Module):
    def __init__(
            self,
            embed_dims,
            backbone,
            head,
            neck,
            decoder=nn.Identity(),
            error_tolerance=1.0,
            depth_num=64,
            opacity_filter=0.05,
            img_h=224,
            img_w=480,
            depth_start=1,
            depth_max=61,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.head = head
        self.neck = neck
        self.decoder = decoder

        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_max = depth_max
        self.gs_render = GaussianRenderer(embed_dims, opacity_filter)
    
        self.error_tolerance = error_tolerance
        self.img_h = img_h
        self.img_w = img_w
        
        bins = self.init_bin_centers()
        self.register_buffer('bins', bins, persistent=False)

    def init_bin_centers(self):
        """
        depth: b d h w
        """
        depth_range = self.depth_max - self.depth_start
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_start
        bin_edges = torch.cumsum(interval, 0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return bin_centers
    
    def pred_depth(self, lidar2img, depth, coords_3d=None):
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            # bins = self.bins * self.bin_scale + self.bin_bias
            coords_3d, coords_d = get_pixel_coords_3d(self.bins, depth, lidar2img, depth_num=self.depth_num, depth_start=self.depth_start, depth_max=self.depth_max, img_h=self.img_h, img_w=self.img_w) # b n w h d 3
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
            
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))
        features, depth, opacities = self.neck(features)

        means3D, cov3D = self.pred_depth(lidar2img, depth)
        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)
        opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)
        
        x, num_gaussians = self.gs_render(features, means3D, cov3D, opacities)
        y = self.decoder(x)
        output = self.head(y)
        output['num_gaussians'] = num_gaussians
        return output
    
class BEVCamera:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=200):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size
        self.image_height = image_size

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        self.set_transform()
    
    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [ 0.,  sh,  0.,         0.],
            [ sw,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [ 0., -sh,  0.,          h/2.],
            [-sw,   0.,  0.,         w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w

class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1
        """ 
        b = features.shape[0]
        device = means3D.device
        
        bev_out = []
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)
        self.set_render_scale(200, 200)
        self.set_Rasterizer(device)
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i][mask[i]],
                opacities=opacities[i][mask[i]],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]
            )
            bev_out.append(rendered_bev)
            
        x = torch.stack(bev_out, dim=0) # b d h w
        num_gaussians = (mask.detach().float().sum(1)).mean().cpu()

        return x, num_gaussians
        
    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device) # self.embed_dims
        # bg_color[-1] = -4
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used 
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)

@torch.no_grad()
def get_pixel_coords_3d(coords_d, depth, lidar2img, img_h=224, img_w=480, depth_num=64, depth_start=1, depth_max=61):
    eps = 1e-5
    
    B, N = lidar2img.shape[:2]
    H, W = depth.shape[-2:]
    scale = img_h // H
    # coords_h = torch.linspace(scale // 2, img_h - scale//2, H, device=depth.device).float()
    # coords_w = torch.linspace(scale // 2, img_w - scale//2, W, device=depth.device).float()
    coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
    coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w
    # coords_d = get_bin_centers(depth_max, depth_start, depth_num).to(depth.device)
    # coords_d = coords_d * bin_scale + bin_bias

    D = coords_d.shape[0]
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    img2lidars = lidar2img.inverse() # b n 4 4

    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

    return coords3d, coords_d

# @torch.no_grad()
def get_bin_centers(max_depth, min_depth, depth_num):
    """
    depth: b d h w
    """
    depth_range = max_depth - min_depth
    interval = depth_range / depth_num
    interval = interval * torch.ones((depth_num+1))
    interval[0] = min_depth
    # interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

    bin_edges = torch.cumsum(interval, 0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers
    