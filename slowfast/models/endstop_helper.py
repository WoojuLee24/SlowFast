import torch.nn as nn
import torch
import torch.nn.functional as F

def get_endstop_function(name, dim_in, dim_out):
    """
    Retrives the Endstopping layer by name
    """
    endstop_funcs = {
        "EndStopping1": EndStopping(dim_in, dim_out,
                                kernel_size=(1, 5, 5), padding=(0, 2, 2), dilation=1, groups=1),
        "EndStopping2": EndStopping2(dim_in, dim_out,
                                 kernel_size=(1, 5, 5), padding=(0, 2, 2), dilation=1, groups=1),
        "DoG": DoG(dim_in, dim_out,
                   kernel_size=(1, 5, 5), padding=(0, 2, 2), dilation=1, groups=1),
        "CompareDog": CompareDoG(dim_in, dim_out,
                                  kernel_size=(1, 5, 5), padding=(0, 2, 2), dilation=1, groups=1),
    }
    assert (
        name in endstop_funcs.keys()
    ), "EndStopping function '{}' not supported".format(name)
    return endstop_funcs[name]


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

    def get_name(self):
        return type(self).__name__

    def forward(self, x):
        x = F.conv3d(x, self.weight, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EndStopping(nn.Module):

    """
    End-stopping kernel for solving aperture problem
    Learnable parameter of pseudo-Difference of Gaussian Kernel
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super(EndStopping, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        self.slope_x = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.slope_y = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.center = self.get_param(self.in_channels, self.out_channels, self.groups)
        self.one = torch.ones([self.out_channels, self.in_channels // self.groups, 1, 1, 1], dtype=torch.float).cuda()
        self.zero = torch.zeros([self.out_channels, self.in_channels // self.groups, 1, 1, 1], dtype=torch.float).cuda()

    def get_param(self, in_channels, out_channels, groups):
        param = torch.zeros([out_channels, in_channels//groups, 1, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('sigmoid'))
        return nn.Parameter(param)

    def get_weight(self, slope_x, slope_y, center, one, zero):

        # bias = 1 / 2 * (torch.sigmoid(center) + one)
        # kernel_x = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one),
        #                       bias - 1 / 2 * (torch.sigmoid(slope_x) + one),
        #                       bias,
        #                       bias - 1 / 2 * (torch.sigmoid(slope_x) + one),
        #                       bias - 2 * 1 / 2 * (torch.sigmoid(slope_x) + one)], dim=3)
        # kernel_x = kernel_x.repeat((1, 1, 1, 1, 5))
        # kernel_y = torch.cat([bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one),
        #                       bias - 1 / 2 * (torch.sigmoid(slope_y) + one),
        #                       bias,
        #                       bias - 1 / 2 * (torch.sigmoid(slope_y) + one),
        #                       bias - 2 * 1 / 2 * (torch.sigmoid(slope_y) + one)], dim=4)
        # kernel_y = kernel_y.repeat((1, 1, 1, 5, 1))
        # kernel = kernel_x + kernel_y

        bias = 1 / 2 * (torch.sigmoid(center) + one)
        sigmoid_slope_x = (torch.sigmoid(slope_x) + one)
        sigmoid_slope_y = (torch.sigmoid(slope_y) + one)
        bias = bias.repeat((1, 1, 1, 5, 5))
        kernel_x = torch.cat([- sigmoid_slope_x,
                              - 1 / 2 * sigmoid_slope_x,
                              zero,
                              - 1 / 2 * sigmoid_slope_x,
                              - sigmoid_slope_x], dim=3)
        kernel_x = kernel_x.repeat((1, 1, 1, 1, 5))
        kernel_y = torch.cat([- sigmoid_slope_y,
                              - 1 / 2 * sigmoid_slope_y,
                              zero,
                              - 1 / 2 * sigmoid_slope_y,
                              - sigmoid_slope_y], dim=4)
        kernel_y = kernel_y.repeat((1, 1, 1, 5, 1))
        kernel = kernel_x + kernel_y + 2 * bias

        return kernel

    def get_name(self):
        return type(self).__name__

    def forward(self, x):
        kernel = self.get_weight(self.slope_x, self.slope_y, self.center, self.one, self.zero)
        x = F.conv3d(x, kernel, stride=1, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CompareDoG(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)

    def get_name(self):
        return type(self).__name__

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EndStopping2(nn.Conv3d):

    """
    End-stopping kernel for solving aperture problem
    Using relu function to learn center-surround suppression
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.groups = groups
        self.groups = in_channels
        self.padding = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        self.param = self.get_param(self.in_channels, self.out_channels, self.kernel_size, self.groups)


    def get_param(self, in_channels, out_channels, kernel_size, groups):
        param = torch.zeros([out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2]], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        return nn.Parameter(param)

    def get_weight(self, param):
        """
        version 1
        center: relu(x)
        surround: - relu(-x)
        """
        center = F.pad(param[:, :, :, 1:4, 1:4], (1, 1, 1, 1))
        surround = param - center
        weight = F.relu(center) - F.relu(surround)
        return weight

    def get_weight2(self, param):
        """
        version 2
        center: relu(x) + relu(-x)
        surround: - relu(x) - relu(-x)
        """
        center = F.pad(param[:, :, :, 1:4, 1:4], (1, 1, 1, 1))
        surround = param - center
        weight = F.relu(center) + F.relu(-center) - F.relu(surround) - F.relu(-surround)
        return weight

    def forward(self, x):
        weight = self.get_weight2(self.param)
        x = F.conv3d(x, weight, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)
        return x
