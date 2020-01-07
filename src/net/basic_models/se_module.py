from torch import nn
import torch
import torch.nn.functional as F

class SELayer3D(nn.Module):
    def __init__(self, channel,temporal, reduction=16):
        super(SELayer3D, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool3d(1)
        #pool have 2 choose, between frame or 1 frame
        self.fc = nn.Sequential(
                nn.Linear(channel * temporal, channel * temporal // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel * temporal // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _, _ = x.size()
        y = nn.AdaptiveAvgPool3d((t,1,1))(x).view(b,c*t)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x + x * y


