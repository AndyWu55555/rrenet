import torch
import torch.nn as nn
import torchvision
import warnings

warnings.filterwarnings('ignore')

g_order = 4


def rot_n(image, k, n):
    angle = (360.0 / n) * k
    return torchvision.transforms.functional.rotate(image, angle)


class RRLConv(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=g_order):
        super(RRLConv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.w = nn.Parameter(torch.empty(c2, c1, k, k))
        # init delta
        self.delta = nn.Parameter(torch.zeros(c2 * g, 1, k, k))
        nn.init.kaiming_uniform_(self.w, a=(5 ** 0.5))

    def build_filters(self):
        rotated_filters = [rot_n(self.w, r, self.g) for r in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, self.c1, self.k, self.k)

    def forward(self, x):
        x = torch.conv2d(
            x,
            self.build_filters() + self.delta,
            stride=self.s,
            padding=self.p,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class PointwiseRREConv(nn.Module):
    def __init__(self, c1, c2, g=g_order):
        super(PointwiseRREConv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = 1
        self.s = 1
        self.p = 0
        self.g = g
        self.pw = nn.Parameter(torch.empty(c2, c1, g, 1, 1))
        nn.init.kaiming_uniform_(self.pw, a=(5 ** 0.5))

    def build_pw_filters(self):
        rotated_filters = []
        for i in range(self.g):
            rotated_filters.append(torch.roll(self.pw, i, dims=-3))
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, self.c1 * self.g, 1, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.conv2d(
            x,
            self.build_pw_filters(),
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class DepthwiseRREConv(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=g_order):
        super(DepthwiseRREConv, self).__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.groups = c2 * g
        self.dw = nn.Parameter(torch.empty(c2, 1, k, k))
        # init delta
        self.delta = nn.Parameter(torch.zeros(c2 * g, 1, k, k))
        nn.init.kaiming_uniform_(self.dw, a=(5 ** 0.5))

    def build_dw_filters(self):
        rotated_filters = [rot_n(self.dw, r, self.g) for r in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, 1, self.k, self.k)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.conv2d(
            x,
            self.build_dw_filters() + self.delta,
            stride=self.s,
            padding=self.p,
            groups=self.groups,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class DepthwiseRREUp(nn.Module):
    def __init__(self, c1, c2, g=g_order):
        super(DepthwiseRREUp, self).__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.k = 2
        self.s = 2
        self.p = 0
        self.g = g
        self.groups = c2 * g
        self.dw = nn.Parameter(torch.empty(c2, 1, self.k, self.k))
        # init delta
        self.delta = nn.Parameter(torch.zeros(c2 * g, 1, self.k, self.k))
        nn.init.kaiming_uniform_(self.dw, a=(5 ** 0.5))

    def build_dw_filters(self):
        rotated_filters = [rot_n(self.dw, r, self.g) for r in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, 1, self.k, self.k)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # using transposed conv for the upsample
        x = torch.conv_transpose2d(
            x,
            self.build_dw_filters() + self.delta,
            stride=self.s,
            padding=self.p,
            groups=self.groups,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class PDRREConv(torch.nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(PDRREConv, self).__init__()
        self.conv = nn.Sequential(
            PointwiseRREConv(c1, c2),
            DepthwiseRREConv(c2, c2, k, s, p)
        )

    def forward(self, x):
        return self.conv(x)


class PDRREUp(torch.nn.Module):
    def __init__(self, c1, c2):
        super(PDRREUp, self).__init__()
        self.up = nn.Sequential(
            PointwiseRREConv(c1, c2),
            DepthwiseRREUp(c2, c2)
        )

    def forward(self, x):
        return self.up(x)


class GMaxPooling(nn.Module):
    def __init__(self, c1, c2):
        super(GMaxPooling, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=2)[0]
