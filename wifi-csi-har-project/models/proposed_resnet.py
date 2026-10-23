import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ProposedCSIResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ProposedCSIResNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1 = ResidualBlock1D(64)
        self.res_block2 = ResidualBlock1D(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out