"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']

# --------------------------------------------------------------
# 2. 모델 구조
# --------------------------------------------------------------
# 1) [make_divisible] 채널 축소 by Width Mult
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 2) [Conv_bn] Kernel:3x3 / Stride:1 or 2 / Padding:1 / ReLU6
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 3) [Conv_bn] Kernel:1x1 / Stride:1 / Padding:0 / ReLU6
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 4) [InvertedResidual Block]
class InvertedResidual(nn.Module):
    # (1) 초기 설정
    def __init__(self, inp, oup, stride, expand_ratio):
        # 0] nn.Module 상속
        super(InvertedResidual, self).__init__()

        # 1] Stride 설정(Stride 형태 고정)
        assert stride in [1, 2]

        # 2] Hidden 설정(Channel 확장 by expand_ratio)
        hidden_dim = round(inp * expand_ratio)

        # 3] Residual Block 설정(Stride 조건, input output 조건)
        self.identity = stride == 1 and inp == oup

        # 4] Inverted Residual Block 생성(Channel + Width Mult)
        # [1] expand_ratio == 1
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # 1]] Depthwise Convolution(dw)
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 2]] PointWise Linear Convolution(pw-linear)
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # [2] expand_ratio != 1
        else:
            self.conv = nn.Sequential(
                # 1]] PointWise Linear Convolution(pw)
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 2]] Depthwise Convolution(dw)
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 3]] PointWise Linear Convolution(pw-linear)
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    # (2) 순전파
    def forward(self, x):
        # 1] Residual Connection인 경우
        if self.identity:
            return x + self.conv(x)
        # 2] Basic Connection인 경우
        else:
            return self.conv(x)

# (1) MobileNetV2 Model 생성
class MobileNetV2(nn.Module):
    # 1] 초기 설정
    def __init__(self, num_classes=1000, width_mult=1.):
        # [0] nn.Module 상속
        super(MobileNetV2, self).__init__()

        # [1] Architecture 구조 설정
        # 1]] Input 설정(Size 형식 고정 + Channel)
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # 2]] Inverted Residual Block 생성(Channel 축소 by Width Mult / 각 Inverted Residual Settings, n개)
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t)) # [첫번째 n] Stride:설정값 / [이후 n] Stride:1
                input_channel = output_channel # [channel 개수] InputChannel 개수 = OutputChannel개수

        # 1]] Output 설정(Channel 축소 by Width Mult)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280

        # 3]] Last Layer 생성 + 통합
        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4]] Classifier 생성
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()

    # 2] 순전파
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print("x : ", x)
        """
        tensor([[ 5.9208e-11,  1.1186e-10, -7.8460e-10,  ...,  1.0627e-09,
         -4.3742e-10, -1.0496e-10],
        [ 5.1078e-10,  1.7102e-10, -2.2938e-09,  ...,  2.6327e-09,
         -8.9200e-10,  1.0217e-11],
        [ 5.3573e-10,  1.0434e-10, -2.2784e-09,  ...,  2.5333e-09,
         -1.1939e-09, -1.8375e-10],
        ...,
        [-7.4871e-10,  2.0639e-10, -1.7850e-09,  ...,  1.4194e-09,
         -1.2905e-09, -3.4316e-10],
        [ 2.1488e-10,  1.7373e-10, -1.5891e-09,  ...,  1.7458e-09,
         -8.2055e-10, -8.8797e-11],
        [ 3.0838e-10,  4.2131e-10, -1.8033e-09,  ...,  1.7994e-09,
         -7.7034e-10, -4.6224e-11]], device='cuda:0')
         """
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# 1. Main문
def mobilenetv2(**kwargs):
    # 1) MobileNetV2 Model 생성
    return MobileNetV2(**kwargs)

