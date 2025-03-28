import timm
from timm.utils.model import freeze

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

class swinT_backbone(nn.Module):
    def __init__(self,model_name='swinv2_cr_tiny_ns_224.sw_in1k',image_height=224,image_width=480,out_indices=[0,2]):
        super().__init__()
        self.model = timm.create_model(model_name,pretrained=True,img_size=(image_height,image_width),features_only=True,out_indices=out_indices)

        dummy = torch.rand(1, 3, image_height, image_width)
        self.output_shapes = [y.shape for y in self.model(dummy)]

    def forward(self,x):
        x = self.model(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, image_height, image_width, out_indices=[2,4], fpn=False, checkpointing=False, freeze_layers=[], embed_dims=256):
        super().__init__()
        self.model = timm.create_model('resnet101', features_only=True, pretrained=True, out_indices=out_indices)

        if len(freeze_layers) !=0:
            freeze(self.model, freeze_layers)

        self.fpn = None
        if fpn:
            channels = [256, 512, 1024, 2048]
            in_channels = [channels[i-1] for i in out_indices]
            self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels, embed_dims)

        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]
        self.output_shapes = output_shapes
        
        if checkpointing:
            self.model.set_grad_checkpointing()
    
    def forward(self,x):
        x = self.model(x)
        if self.fpn is not None:
            _in = OrderedDict()
            for i,tmp_x in enumerate(x):
                _in[i] = tmp_x
            x = self.fpn(_in)
            x = [v for _, v in x.items()]
        return x
    
class TimmBackbone(nn.Module):
    def __init__(self, model_name, image_height, image_width, out_indices=[2,4], fpn=False, checkpointing=False, freeze_layers=[], embed_dims=256):
        """
        model_name: resnet50, resnet101, efficientnet_b4, ...
        """
        super().__init__()
        self.model = timm.create_model(model_name, features_only=True, pretrained=True, out_indices=out_indices)
        if model_name == 'efficientnet_b4':
            if out_indices[-1] == 3:
                del self.model.blocks[6]
                del self.model.blocks[5]
        
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self.model(dummy)]

        self.fpn = None
        if fpn:
            channels = [shape[1] for shape in output_shapes]
            self.fpn = torchvision.ops.FeaturePyramidNetwork(channels, embed_dims)
            output_shapes = [x.shape for x in self(dummy)]
            
        self.output_shapes = output_shapes
        
        if checkpointing:
            self.model.set_grad_checkpointing()

        if len(freeze_layers) !=0:
            freeze(self.model, freeze_layers)
    
    def forward(self,x):
        x = self.model(x)

        if self.fpn is not None:
            _in = OrderedDict()
            for i,tmp_x in enumerate(x):
                _in[i] = tmp_x
            x = self.fpn(_in)
            x = [v for _, v in x.items()]

        return x
    
class SwinB(nn.Module):
    def __init__(self, image_height, image_width, out_indices=[2,4]):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', features_only=True, pretrained=True, out_indices=out_indices, img_size=[image_height,image_width])

    def forward(self, x):
        x = self.model(x)
        x = [_.permute(0,3,1,2) for _ in x]
        return x

if __name__ == '__main__':
    swinT = swinT_backbone(model_name='vit_small_patch16_384')
    print(swinT.output_shapes)