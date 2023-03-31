import torch.nn as nn
import math

# 2. 모델 구조
# 1) [Conv_bn] Kernel:3x3 / Stride:1 or 2 / Padding:1 / ReLU6
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 2) [Conv_bn] Kernel:1x1 / Stride:1 / Padding:0 / ReLU6
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 3) [make_divisible] 채널 축소 by Width Mult
def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by) # 굳이 divisible_by를 나눠야하나? 결국 똑같지 않나???

# 4) [InvertedResidual Block]
class InvertedResidual(nn.Module):
    # (1) 초기 설정
    def __init__(self, inp, oup, stride, expand_ratio):
        # 0] nn.Module 상속
        super(InvertedResidual, self).__init__()

        # 1] Stride 설정(Stride 형태 고정)
        self.stride = stride
        assert stride in [1, 2]

        # 2] Hidden 설정(Channel 확장 by expand_ratio)
        hidden_dim = int(inp * expand_ratio)

        # 3] Residual Block 설정(Stride 조건, input output 조건)
        self.use_res_connect = self.stride == 1 and inp == oup

        # 3] Inverted Residual Block 생성(Channel + Width Mult)
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
        if self.use_res_connect:
            return x + self.conv(x)
        # 2] Basic Connection인 경우
        else:
            return self.conv(x)

# (1) MobileNetV2 Model 생성
class MobileNetV2(nn.Module):
    # 1] 초기 설정
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        # [0] nn.Module 상속
        super(MobileNetV2, self).__init__()

        # [1] Architecture 구조 설정
        # 0]] Input 설정(Channel) + Output 설정(Channel) + Inverted Residual
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        last_channel = 1280

        # 1]] Input 설정(Size 형식 고정 + Channel) + Output 설정(Channel 축소 by Width Mult)
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.features = [conv_bn(3, input_channel, 2)]
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # 2]] Inverted Residual Block 생성(Channel 축소 by Width Mult / 각 Inverted Residual Settings, n개)
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:                      # [첫번째 n] Stride:설정값
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:                           # [이후 n] Stride:1
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel  # [channel 개수] InputChannel 개수 = OutputChannel개수

        # 3]] Last Layer 생성 + 통합
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 4]] Classifier 생성
        self.classifier = nn.Linear(self.last_channel, n_class)
        self._initialize_weights()

    # 2] 순전파
    def forward(self, x):
        x = self.features(x)
        print(x)
        x = x.mean(3).mean(2)
        print(x)
        x = self.classifier(x)
        print(x)
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# 1) MobileNetV2 Model 생성
def mobilenet_v2(pretrained=True):
    # (1) MobileNetV2 Model 생성
    model = MobileNetV2(width_mult=1)

    # (2) 사전학습된 MobileNetV2 Model 불러오기
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url('https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model

# 1. Main문
if __name__ == '__main__':
    # 1) MobileNetV2 Model 생성
    net = mobilenet_v2(True)





