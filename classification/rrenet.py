import torch
import torch.nn as nn
import warnings

from torchvision.transforms import v2

warnings.filterwarnings('ignore')

params = {
    'g_order': 4,
    'rre': True,
    'std': 0,
    'default_act': nn.SiLU
}


def rot_n(x, i, g):
    if g in [2, 4]:
        return torch.rot90(x, 4 // params['g_order'] * i, dims=[-2, -1])
    else:
        # Affine transformation
        if len(x.shape) == 5:
            x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
            angle = (360.0 / g) * i
            x = v2.functional.rotate(x, angle)
            return x.view(x.shape[0], -1, params['g_order'], x.shape[-2], x.shape[-1])
        else:
            angle = (360.0 / g) * i
            return v2.functional.rotate(x, angle)


class RRLConv(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(RRLConv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = params['g_order']
        self.rre = params['rre']
        self.w = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.w, a=(5 ** 0.5))
        if self.rre:
            self.g_bias = nn.Parameter(torch.zeros(c2, self.g, 1, k, k), requires_grad=True)
            if params['std'] != 0:
                torch.nn.init.normal_(self.g_bias, mean=0.0, std=params['std'])

    def build_filters(self):
        rotated_filters = [rot_n(self.w, r, self.g) for r in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        if self.rre:
            rotated_filters += self.g_bias
        return rotated_filters.view(self.c2 * self.g, self.c1, self.k, self.k)

    def forward(self, x):
        x = torch.conv2d(
            x,
            self.build_filters(),
            stride=self.s,
            padding=self.p,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class DepthwiseRREConv(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(DepthwiseRREConv, self).__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = params['g_order']
        self.groups = c2 * self.g
        self.rre = params['rre']
        self.dw = nn.Parameter(torch.empty(c2, 1, k, k))
        nn.init.kaiming_uniform_(self.dw, a=(5 ** 0.5))
        if self.rre:
            self.g_bias = nn.Parameter(torch.zeros(c2, self.g, 1, k, k))
            if params['std'] != 0:
                torch.nn.init.normal_(self.g_bias, mean=0.0, std=params['std'])

    def build_dw_filters(self):
        rotated_filters = [rot_n(self.dw, r, self.g) for r in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        if self.rre:
            rotated_filters += self.g_bias
        return rotated_filters.view(self.c2 * self.g, 1, self.k, self.k)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.conv2d(
            x,
            self.build_dw_filters(),
            stride=self.s,
            padding=self.p,
            groups=self.groups,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class PointwiseRREConv(nn.Module):
    def __init__(self, c1, c2):
        super(PointwiseRREConv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = 1
        self.s = 1
        self.p = 0
        self.g = params['g_order']
        self.pw = nn.Parameter(torch.empty(c2, c1, self.g, 1, 1))
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


class DepthwiseRREUp(nn.Module):
    def __init__(self, c1, c2):
        super(DepthwiseRREUp, self).__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.k = 2
        self.s = 2
        self.p = 0
        self.g = params['g_order']
        self.groups = c2 * params['g_order']
        self.dw = nn.Parameter(torch.empty(c2, 1, self.k, self.k))
        # init delta
        self.delta = nn.Parameter(torch.zeros(c2 * self.g, 1, self.k, self.k))
        nn.init.kaiming_uniform_(self.dw, a=(5 ** 0.5))

    def build_dw_filters(self):
        rotated_filters = [rot_n(self.dw, i, self.g) for i in range(self.g)]
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


class GMaxPooling(nn.Module):
    def __init__(self, c1, c2):
        super(GMaxPooling, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=2)[0]


class RRLCBA(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(RRLCBA, self).__init__()
        self.conv = nn.Sequential(
            RRLConv(c1, c2, k, s, p),
            nn.BatchNorm3d(c2),
            params['default_act']()
        )

    def forward(self, x):
        return self.conv(x)


class PDRRECBA(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(PDRRECBA, self).__init__()
        self.conv = nn.Sequential(
            PDRREConv(c1, c2, k, s, p),
            nn.BatchNorm3d(c2),
            params['default_act']()
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseRRECBA(nn.Module):
    def __init__(self, c1, c2):
        super(PointwiseRRECBA, self).__init__()
        self.conv = nn.Sequential(
            PointwiseRREConv(c1, c2),
            nn.BatchNorm3d(c2),
            params['default_act']()
        )

    def forward(self, x):
        return self.conv(x)


class _Bottleneck(nn.Module):
    def __init__(self, c):
        super(_Bottleneck, self).__init__()
        self.m = nn.Sequential(
            PDRRECBA(c, c, 3, 1, 1),
            PDRRECBA(c, c, 3, 1, 1)
        )

    def forward(self, x):
        return self.m(x) + x


class PDRREBlock(nn.Module):
    def __init__(self, c1, c2, n=1):
        super(PDRREBlock, self).__init__()
        c_ = int(c2 * 0.75)
        self.conv1 = PointwiseRRECBA(c1, c_)
        self.m = nn.Sequential(
            *[_Bottleneck(c_) for _ in range(n)]
        )
        self.conv2 = PointwiseRRECBA(c_, c2)
        self.added = c1 == c2

    def forward(self, x):
        if self.added:
            y = self.conv1(x)
            y = self.m(y)
            return self.conv2(y) + x
        else:
            x = self.conv1(x)
            x = self.m(x)
            return self.conv2(x)

model_size = {
    'n': {'chs': [16, 32, 64, 128], 'blocks': [1, 2, 2, 1]},
    's': {'chs': [32, 64, 128, 256], 'blocks': [1, 2, 2, 1]},
    'm': {'chs': [48, 96, 192, 384], 'blocks': [2, 4, 4, 2]},
}


class _RRENet(nn.Module):
    def __init__(self, size, num_classes=100):
        super(_RRENet, self).__init__()
        assert size in ['n', 's', 'm']
        chs = model_size[size]['chs']
        blocks = model_size[size]['blocks']
        self.lift = RRLCBA(3, chs[0], 3, 1, 1)
        self.m = nn.Sequential(
            PDRRECBA(chs[0], chs[1], 3, 1, 1),
            PDRREBlock(chs[1], chs[1], n=blocks[0]),
            PDRRECBA(chs[1], chs[2], 2, 2, 0),
            PDRREBlock(chs[2], chs[2], n=blocks[1]),
            PDRRECBA(chs[2], chs[3], 2, 2, 0),
            PDRREBlock(chs[3], chs[3], n=blocks[2]),
            PDRRECBA(chs[3], chs[3], 3, 1, 1),
            PDRREBlock(chs[3], chs[3], n=blocks[3]),
            GMaxPooling(chs[3], chs[3])
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(chs[3], num_classes)

    def forward(self, x):
        x = self.lift(x)
        x = self.m(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_rrenet(size='n', g_order=4, rre=True, std=0, num_classes=100):
    params['g_order'] = g_order
    params['rre'] = rre
    params['std'] = std
    return _RRENet(size, num_classes)
