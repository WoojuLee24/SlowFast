import torch.nn as nn
import torch
import torch.nn.functional as F


def get_endstop_function(name, dim_in, dim_out, stride=[1, 1, 1], dilation=[1, 1, 1], groups=1):
    """
    Retrives the Endstopping layer by name
    """
    endstop_funcs = {
        "EndStoppingDivide": EndStoppingDivide,
        "EndStoppingDilation": EndStoppingDilation,
        "CompareDog": CompareDoG,
    }
    assert (
        name in endstop_funcs.keys()
    ), "EndStopping function '{}' not supported".format(name)
    return endstop_funcs[name]


class EndStoppingDivide(nn.Conv3d):

    """
    End-stopping Divide kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2]], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight_5x5(self, param):
        """
        5x5 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, :, 1:4, 1:4], (1, 1, 1, 1))
        surround = param - center
        surround = surround * 9/16
        weight = F.relu(center) + F.relu(-center) - F.relu(surround) - F.relu(-surround)
        return weight

    def get_weight_3x3(self, param):
        """
        3x3 surround modulation
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround /= 8
        weight = F.relu(center) + F.relu(-center) - F.relu(surround) - F.relu(-surround)
        return weight

    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def forward(self, x):
        weight = self.get_weight_3x3(self.param)
        x = F.conv3d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return x


class EndStoppingDilation(nn.Conv3d):

    """
    End-stopping dilation kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param1 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)
        self.param2 = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2]], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_center(self, param):
        center = F.relu(param) + F.relu(-param)
        return center

    def get_surround(self, param):
        center = F.pad(param[:, :, :, 1:2, 1:2], (1, 1, 1, 1))
        surround = param - center
        surround = -F.relu(surround) - F.relu(-surround)
        return surround

    def forward(self, x):
        center = self.get_center(self.param1)
        surround = self.get_surround(self.param2)
        x1 = F.conv3d(x, center, stride=self.stride, dilation=1, padding=self.padding, groups=self.groups)
        x2 = F.conv3d(x, surround, stride=self.stride, dilation=2, padding=(0, 2, 2), groups=self.groups)
        x = x1 + x2
        return x



class CompareDoG(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)

    def get_name(self):
        return type(self).__name__

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

