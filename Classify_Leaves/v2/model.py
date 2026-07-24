"""
model.py
----------------------------------------
Custom ResNet for Leaf Classification
Author: ChatGPT

Features:
- Custom ResNet18-like architecture
- Kaiming initialization
- Compatible with train.py
"""

import torch
import torch.nn as nn


# ==========================================
# Residual Block
# ==========================================
class Residual(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_1x1conv=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        y = y + self.shortcut(x)

        return self.relu(y)


# ==========================================
# Build ResNet Stage
# ==========================================
def resnet_block(
        in_channels,
        out_channels,
        num_residuals,
        first_block=False):

    layers = []

    for i in range(num_residuals):

        if i == 0 and not first_block:

            layers.append(
                Residual(
                    in_channels,
                    out_channels,
                    stride=2,
                    use_1x1conv=True
                )
            )

        else:

            layers.append(
                Residual(
                    out_channels if i > 0 else in_channels,
                    out_channels
                )
            )

    return nn.Sequential(*layers)


# ==========================================
# ResNet18
# ==========================================
class ResNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.layer1 = resnet_block(
            64,
            64,
            2,
            first_block=True
        )

        self.layer2 = resnet_block(
            64,
            128,
            2
        )

        self.layer3 = resnet_block(
            128,
            256,
            2
        )

        self.layer4 = resnet_block(
            256,
            512,
            2
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x

    # -------------------------------------
    # Weight Initialization
    # -------------------------------------
    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)


# ==========================================
# Create Model
# ==========================================
def get_model(num_classes):

    return ResNet(num_classes)


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    model = get_model(num_classes=176)

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    print(model)

    print("Output shape:", y.shape)