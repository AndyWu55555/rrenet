from .rreconv import *

default_act = nn.SiLU


class RRLCBA(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(RRLCBA, self).__init__()
        self.conv = nn.Sequential(
            RRLConv(c1, c2, k, s, p),
            nn.BatchNorm3d(c2),
            default_act()
        )

    def forward(self, x):
        return self.conv(x)


class PDRRECBA(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(PDRRECBA, self).__init__()
        self.conv = nn.Sequential(
            PDRREConv(c1, c2, k, s, p),
            nn.BatchNorm3d(c2),
            default_act()
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseRRECBA(nn.Module):
    def __init__(self, c1, c2):
        super(PointwiseRRECBA, self).__init__()
        self.conv = nn.Sequential(
            PointwiseRREConv(c1, c2),
            nn.BatchNorm3d(c2),
            default_act()
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


class _BaseMaxPooling(nn.Module):
    def __init__(self, k, s, p, g=g_order):
        super(_BaseMaxPooling, self).__init__()
        self.g = g
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.m(x)
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class GSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super(GSPPF, self).__init__()
        c_ = c1 // 2
        self.conv1 = PointwiseRRECBA(c1, c_)
        self.m = _BaseMaxPooling(k, 1, k // 2)
        self.conv2 = PointwiseRRECBA(c_ * 4, c2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
