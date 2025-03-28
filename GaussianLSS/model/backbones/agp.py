from typing import Iterable, Optional
import torch
import torch.nn as nn

class AlignRes(nn.Module):
    """Align resolutions of the outputs of the backbone."""

    def __init__(
        self,
        mode="upsample",
        scale_factors: Iterable[int] = [1, 2],
        in_channels: Iterable[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if mode == "upsample":
            for s in scale_factors:
                if s != 1:
                    self.layers.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        elif mode == "conv2dtranspose":
            for i, in_c in enumerate(in_channels):
                if scale_factors[i] != 1:
                    self.layers.append(
                        nn.ConvTranspose2d(
                            in_c, in_c, kernel_size=2, stride=2, padding=0
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        else:
            raise NotImplementedError
        return

    def forward(self, x):
        return [self.layers[i](xi) for i, xi in enumerate(x)]


class PrepareChannel(nn.Module):
    """Transform the feature map to align with Network."""

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        interm_c=128,
        out_c: Optional[int] = 128,
        tail_mode="identity",
        depth_num=0,
    ):
        super().__init__()
        assert tail_mode in ["identity", "conv2d"]
        assert depth_num != 0

        in_c = sum(in_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
        )
        self.depth_layers = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
        )
        self.opacity_layers = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, 1, kernel_size=1, padding=0)
        )

        if tail_mode == "identity":
            self.tail = nn.Identity()
        elif tail_mode == "conv2d":
            # Used in SimpleBEV
            self.tail = nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0)
        
    def forward(self, x):
        return self.tail(self.layers(x)), self.depth_layers(x), self.opacity_layers(x).sigmoid()

class AGPNeck(nn.Module):
    """
    Upsample outputs of the backbones, group them and align them to be compatible with Network.

    Note: mimics UpsamplingConcat in SimpleBEV.
    """

    def __init__(
        self,
        align_res_layer,
        prepare_c_layer,
        group_method=lambda x: torch.cat(x, dim=1),
        list_output=False,
    ):
        """
        Args:
            - align_res_layer: upsample layers at different resolution to the same.
            - group_method: how to gather the upsampled layers.
            - prepare_c_layer: change the channels of the upsampled layers in order to align with the network.
        """
        super().__init__()

        self.align_res_layer = align_res_layer
        self.group_method = group_method
        self.prepare_c_layer = prepare_c_layer
        self.list_output = list_output

    def forward(self, x: Iterable[torch.Tensor]):
        if x[0].ndim == 5:
            x = [y.flatten(0,1) for y in x]
        
        # Align resolution of inputs.
        x = self.align_res_layer(x)

        # Group inputs.
        x = self.group_method(x)

        # Change channels of final input.
        x, depth, opacity = self.prepare_c_layer(x)
        if self.list_output:
            x = [x]
        return x, depth, opacity