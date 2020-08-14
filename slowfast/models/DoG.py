import torch.nn as nn
import torch
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill

class DoG(nn.Conv3d):

    """
    Fixed Difference of Gaussian
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        self.weight = self.get_weight5x5(self.in_channels, self.out_channels, self.groups)
        self.weight.requires_grad = False

    def get_weight5x5(self, in_channels, out_channels, groups):
        # kernel 5x5
        kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27], [-0.23, 0.17, 0.49, 0.17, -0.23],
                                 [-0.18, 0.49, 1, 0.49, -0.18], [-0.23, 0.17, 0.49, 0.17, -0.23],
                                 [-0.27, -0.23, -0.18, -0.23, -0.27]])
        kernel = kernel.repeat(in_channels // groups, in_channels // groups, 1, 1, 1)
        kernel = kernel.to(dtype=torch.float)
        kernel = kernel.cuda()
        return nn.Parameter(kernel)

    def forward(self, x):
        x = F.conv3d(x, self.weight, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EndStopping(nn.Conv3d):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.weight = self.get_weight(self.slope_x, self.slope_y, self.center)

    def get_param(self, in_channels, out_channels, groups):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1, 1], dtype=torch.float)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('sigmoid'))
        return param

    def get_weight(self, slope_x, slope_y, center):
        one = torch.ones([self.out_channels, self.in_channels // self.groups, 1, 1, 1], dtype=torch.float).cuda()
        bias = 1 / 2 * (torch.sigmoid(center) + one)
        kernel_x = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one),
                              bias - 1 / 2 * (torch.sigmoid(slope_x) + one),
                              bias,
                              bias - 1 / 2 * (torch.sigmoid(slope_x) + one),
                              bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one)], dim=3)
        kernel_x = kernel_x.repeat((1, 1, 1, 1, 5))
        kernel_y = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one),
                              bias - 1 / 2 * (torch.sigmoid(slope_y) + one),
                              bias,
                              bias - 1 / 2 * (torch.sigmoid(slope_y) + one),
                              bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one)], dim=4)
        kernel_y = kernel_y.repeat((1, 1, 1, 5, 1))
        kernel = kernel_x + kernel_y
        return nn.Parameter(kernel)

    def forward(self, x):
        x = F.conv3d(x, self.weight, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)
        return x


