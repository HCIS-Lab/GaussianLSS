import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

BottleneckBlock = lambda x: Bottleneck(x, x//4)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BottleneckBlock(out_channels),
            BottleneckBlock(out_channels),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, inC=128, outC=128):
        super(BevEncode, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inC, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.enc1 = nn.Sequential(
            BottleneckBlock(128),
            BottleneckBlock(128),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BottleneckBlock(128),
            BottleneckBlock(128),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            BottleneckBlock(256),
            BottleneckBlock(256),
        )

        self.up1 = Up(256 + 128, 128)  
        self.up2 = Up(128 + 128, 128)  
        self.up3 = nn.Sequential( 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv1(x)  
        x1 = self.enc1(x)  
        x2 = self.enc2(x1) 
        x3 = self.enc3(x2)

        x = self.up1(x3, x2) 
        x = self.up2(x, x1)  
        x = self.up3(x)      

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
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_last, stop-start, 1)
            )
            self.to_logits = nn.ModuleDict(layer_dict)
        else:
            self.to_logits = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_last, dim_max, 1)
            )

    def forward(self, x):
        if self.multi_head:
            return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}
