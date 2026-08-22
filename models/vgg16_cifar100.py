"""Standard VGG-16-BN for CIFAR-100 (32x32 input)."""
import torch.nn as nn


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG16_CIFAR100(nn.Module):
    """VGG-16 with BatchNorm, adapted for CIFAR-100.
    5 maxpools on 32x32 -> 1x1x512 -> Linear(512, 100).
    """
    def __init__(self, num_classes=100):
        super().__init__()
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)])
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_features(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)
