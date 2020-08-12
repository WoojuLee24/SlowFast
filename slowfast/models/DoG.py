import torch.nn as nn
import torch
import torch.nn.functional as F

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
    Learnable paramter
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels=in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        # self.sig = torch.


    def forward(self, x):
        x = F.conv3d(x, self.weight, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)