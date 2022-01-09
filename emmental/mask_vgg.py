import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from emmental.modules import MaskedLinear, MaskedConv2d
from typing import cast, List
import torchvision.models as models
from torchvision.models.vgg import model_urls


def create_masked_linear(in_features, out_features, pruning_method, mask_scale):
    ret = MaskedLinear(in_features=in_features,
                       out_features=out_features,
                       pruning_method=pruning_method,
                       mask_scale=mask_scale)

    return ret


class MaskedSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input_, threshold, mask_state):
        for module in self:
            if isinstance(module, MaskedLinear) or isinstance(module, MaskedConv2d):
                input_ = module(input_, threshold, mask_state)
            else:
                input_ = module(input_)
        return input_


class MaskedVGG(models.VGG):

    def __init__(self, pruning_method, mask_scale, features: nn.Module, num_classes: int = 1000, init_weights: bool = True) -> None:
        super().__init__(features, num_classes=num_classes, init_weights=init_weights)

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = MaskedSequential(
            create_masked_linear(512 * 7 * 7, 4096, pruning_method, mask_scale),
            nn.ReLU(True),
            nn.Dropout(),
            create_masked_linear(4096, 4096, pruning_method, mask_scale),
            nn.ReLU(True),
            nn.Dropout(),
            create_masked_linear(4096, num_classes, pruning_method, mask_scale)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, threshold: float, mask_state: int) -> torch.Tensor:

        x = self.features(x, threshold, mask_state)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x, threshold, mask_state)

        return x


class MaskedVGG16(MaskedVGG):
    r'''Masked version of VGG16. The masking scheme is inspired by huggingface/block_movement_pruning.
       This model generate binary masks based on the pruning method on the fly. Note that we add the 
       pruning threshold to the forward function.

       Parameters:
            pruning_method: indicates pruning means
                "topK", "threshold", "sigmoied_threshold", "magnitude", "l0"
            mask_scale: initial value of the mask, 0 by default
    '''

    def __init__(self, pruning_method: str = "magnitude", mask_scale: int = 0, num_classes: int = 1000,
                 init_weights: bool = True, batch_norm: bool = False, pretrained: bool = False) -> None:

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        layers: List[nn.Module] = []

        # According to the paper, the first layer should not be pruned
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        in_channels = 64

        # Create the following layers in the mask version
        for v in cfg[1:]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                # conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, pruning_method=pruning_method, mask_scale=mask_scale)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        features = MaskedSequential(*layers)
        super().__init__(pruning_method=pruning_method, mask_scale=mask_scale, features=features, 
                         num_classes=num_classes, init_weights=init_weights)

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["vgg16"])
            self.load_state_dict(state_dict)
