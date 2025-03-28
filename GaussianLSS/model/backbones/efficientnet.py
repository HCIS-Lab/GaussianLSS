import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet as effmodel
    
class EfficientNet(nn.Module):
    def __init__(self, image_height, image_width, return_list=["reduction_3", "reduction_4"], version="b4", checkpoint=False):
        super().__init__()

        self.version = version
        self.checkpoint = checkpoint
        self.return_list = return_list
        self._init_efficientnet(version)

        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy, True)]

        self.output_shapes = output_shapes
        
    def _init_efficientnet(self, version):
        trunk = effmodel.from_pretrained(f"efficientnet-{version}")

        self._conv_stem, self._bn0, self._swish = (
            trunk._conv_stem,
            trunk._bn0,
            trunk._swish,
        )
        self.drop_connect_rate = trunk._global_params.drop_connect_rate

        self._blocks = nn.ModuleList()
        for idx, block in enumerate(trunk._blocks):
            if version == "b0" and idx > 10 or version == "b4" and idx > 21:
                break
            self._blocks.append(block)

        del trunk

    def forward(self, x, return_all=False):
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            if self.training and self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, drop_connect_rate)
            else:
                x = block(x, drop_connect_rate=drop_connect_rate)
            # x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints)+1}"] = prev_x
            prev_x = x

            if self.version == "b0" and idx == 10:
                break
            if self.version == "b4" and idx == 21:
                break

        # Head
        endpoints[f"reduction_{len(endpoints)+1}"] = x

        # if not return_all:
        #     list_keys = ["reduction_3", "reduction_4"]
        # else:
        #     list_keys = list(endpoints.keys())
        list_keys = self.return_list

        return [endpoints[k] for k in list_keys]
