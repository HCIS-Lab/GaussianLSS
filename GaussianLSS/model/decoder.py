import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, in_channels):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # print(x.shape) # torch.Size([1, 128, 200, 200])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape) # torch.Size([1, 64, 100, 100])
        x1 = self.layer1(x)
        # print(x1.shape) # torch.Size([1, 64, 100, 100])
        x = self.layer2(x1)
        # print(x.shape) # torch.Size([1, 128, 50, 50])
        x = self.layer3(x)
        # print(x.shape) # torch.Size([1, 256, 25, 25])

        x = self.up1(x, x1)
        # print(x.shape) # torch.Size([1, 256, 100, 100])
        x = self.up2(x)
        # print(x.shape) # torch.Size([1, 128, 200, 200])

        return x
    
class SegHead(nn.Module):
    def __init__(self, 
            dim_last, 
            multi_head, 
            outputs,
        ):
        super().__init__()

        self.multi_head = multi_head
        self.outputs = outputs

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        if multi_head:
            layer_dict = {}
            for k, (start, stop) in outputs.items():
                layer_dict[k] = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.InstanceNorm2d(dim_last),
                nn.GELU(),
                nn.Conv2d(dim_last, stop-start, 1)
            )
            self.to_logits = nn.ModuleDict(layer_dict)
        else:
            self.to_logits = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.InstanceNorm2d(dim_last),
                nn.GELU(),
                nn.Conv2d(dim_last, dim_max, 1)
            )

    def forward(self, x, aux=False):
        if self.multi_head:
            if aux:
                return {'VEHICLE': self.to_logits['VEHICLE'](x)}
            else:
                return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}
