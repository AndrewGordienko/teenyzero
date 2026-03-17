import torch
import torch.nn as nn
import torch.nn.functional as F

from teenyzero.alphazero.config import (
    INPUT_PLANES,
    MODEL_CHANNELS,
    MODEL_RES_BLOCKS,
    POLICY_HEAD_CHANNELS,
    VALUE_HEAD_HIDDEN,
)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaNet(nn.Module):
    def __init__(
        self,
        input_planes=INPUT_PLANES,
        num_res_blocks=MODEL_RES_BLOCKS,
        channels=MODEL_CHANNELS,
        policy_head_channels=POLICY_HEAD_CHANNELS,
        value_hidden=VALUE_HEAD_HIDDEN,
    ):
        super().__init__()

        self.input_planes = input_planes
        self.num_res_blocks = num_res_blocks
        self.channels = channels

        self.conv_in = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])

        self.pol_conv = nn.Conv2d(channels, policy_head_channels, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(policy_head_channels)
        self.pol_fc = nn.Linear(policy_head_channels * 8 * 8, 4672)

        self.val_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(1 * 8 * 8, value_hidden)
        self.val_fc2 = nn.Linear(value_hidden, 1)

        nn.init.zeros_(self.val_fc2.weight)
        nn.init.zeros_(self.val_fc2.bias)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.pol_fc(p)

        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v))

        return p, v
