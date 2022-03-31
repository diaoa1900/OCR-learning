import paddle
from functools import partial
from typing import List


f_four = []


def make_divisible(channel, divisor=8, mini_channel=None):
    # 确保所有层都是8的倍数
    if mini_channel is None:
        mini_channel = divisor
    new_channel = max(mini_channel, int(channel + divisor / 2) // divisor * divisor)
    if new_channel < 0.9 * channel:
        new_channel += divisor
    return new_channel


class ConvBNActivation(paddle.nn.Sequential):
    # 卷积+BN+激活
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        if activation_layer is None:
            activation_layer = paddle.nn.ReLU6
        super(ConvBNActivation, self).__init__(paddle.nn.Conv2D(in_channels=in_planes,
                                                                out_channels=out_planes,
                                                                kernel_size=kernel_size,
                                                                stride=stride,
                                                                padding=padding,
                                                                groups=groups,
                                                                bias_attr=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SequeezeExcitation(paddle.nn.Layer):
    # SE模块
    def __init__(self, input_channel: int, squeeze_factor: int = 4):
        super(SequeezeExcitation, self).__init__()
        squeeze_channel = make_divisible(input_channel // squeeze_factor, 8)
        self.fc1 = paddle.nn.Conv2D(input_channel, squeeze_channel, 1)
        self.fc2 = paddle.nn.Conv2D(squeeze_channel, input_channel, 1)

    def forward(self, x):
        scale = paddle.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = paddle.nn.functional.relu(scale)
        scale = self.fc2(scale)
        scale = paddle.nn.functional.hardsigmoid(scale)
        return scale * x


class InvertResidualConfig:
    def __init__(self, input_channel, kernel_size, expand_channel, output_channel, use_se, activation, stride,
                 width_multi):
        self.input_channel = self.adjust_channel(input_channel, width_multi)
        self.kernel_size = kernel_size
        self.expand_channel = self.adjust_channel(expand_channel, width_multi)
        self.output_channel = self.adjust_channel(output_channel, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channel(inp, multi):
        return make_divisible(inp * multi, 8)


class InvertResidual(paddle.nn.Layer):
    # 倒残差结构
    def __init__(self, config: InvertResidualConfig, norm_layer, scale):
        super(InvertResidual, self).__init__()

        if config.stride not in [1, 2]:
            raise ValueError("Illegal stride value!")

        self.use_connect = (config.stride == 1 and config.input_channel == config.output_channel)

        self.feature_need = config.stride == 2

        activation_layer = paddle.nn.Hardsigmoid if config.use_hs else paddle.nn.ReLU

        layers = []

        # 1x1升维
        if config.input_channel != config.expand_channel:
            layers.append(ConvBNActivation(make_divisible(config.input_channel*scale), make_divisible(config.expand_channel*scale),
                                           kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        # depthwise部分(group和输入channel相等，卷积核和输入channel一一对应去运算)
        layers.append(ConvBNActivation(make_divisible(config.expand_channel*scale), make_divisible(config.expand_channel*scale),
                                       kernel_size=config.kernel_size,
                                       stride=config.stride, groups=make_divisible(config.expand_channel*scale),
                                       norm_layer=norm_layer, activation_layer=activation_layer))

        # 是否使用SE模块
        if config.use_se:
            layers.append(SequeezeExcitation(make_divisible(config.expand_channel*scale)))

        # 1x1降维
        layers.append(ConvBNActivation(make_divisible(config.expand_channel*scale), make_divisible(config.output_channel*scale),
                                       kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=None))

        self.block = paddle.nn.Sequential(*layers)
        self.output_channel = make_divisible(config.output_channel*scale)

    def forward(self, x):
        global f_four
    # 将缩小尺寸前的最后一个特征图加入到list中
        if self.feature_need:
            f_four.append(x)
        result = self.block(x)
        # 是否使用shortcut
        if self.use_connect:
            result += x
        """if self.feature_need:
            f_four.append(result)"""
        return result


class MobNetV3(paddle.nn.Layer):
    def __init__(self, InvertResidualSettings: List[InvertResidualConfig], last_channel: int,
                 num_classes: int = 1000, scale=0.5, block=None, normal_layer=None):
        super(MobNetV3, self).__init__()
        self.feature_four = None
        self.out_channels = []

        if not InvertResidualSettings:
            raise ValueError("InvertResidualSettings must be not empty!")
        elif not isinstance(InvertResidualSettings, List):
            raise TypeError("InvertResidualSettings must be List!")
        elif not all([isinstance(s, InvertResidualConfig) for s in InvertResidualSettings]):
            raise TypeError("InvertResidualSettings must be InvertResidualConfig!")

        if block is None:
            block = InvertResidual
        if normal_layer is None:
            normal_layer = partial(paddle.nn.BatchNorm2D, epsilon=0.001, momentum=0.01)

        layers: List[paddle.nn.Layer] = []
    # 第一层卷积
        layers.append(ConvBNActivation(3, make_divisible(16*scale), kernel_size=3, stride=2, norm_layer=normal_layer,
                                       activation_layer=paddle.nn.Hardswish))

        for irs in InvertResidualSettings:
            layers.append(block(irs, normal_layer, scale))
        last_conv_input_channel_before_fc = make_divisible(InvertResidualSettings[-1].output_channel*scale)
        last_conv_output_channel_before_fc = last_conv_input_channel_before_fc * 6
        layers.append(
            ConvBNActivation(last_conv_input_channel_before_fc, last_conv_output_channel_before_fc, kernel_size=1,
                             norm_layer=normal_layer, activation_layer=paddle.nn.Hardswish))

        self.features = paddle.nn.Sequential(*layers)
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(1)
        self.classifier1 = paddle.nn.Sequential(paddle.nn.Linear(last_conv_output_channel_before_fc, last_channel),
                                                paddle.nn.Hardswish())
        self.classifier2 = paddle.nn.Linear(last_channel, num_classes)

    def forward(self, x):
        result = self.features(x)
        global f_four
        f_four.append(result)
        if len(f_four) > 4:
            f_four = f_four[-4:]
        print("提取到的特征shape为")
        print(result.shape)
        result = self.avgpool(result)
        result = paddle.flatten(result, 1)
        result = self.classifier1(result)
        result = self.classifier2(result)
        self.feature_four = f_four
        for j in f_four:
            self.out_channels.append(j.shape[1])
        # return result
        return self.feature_four


def MobileNet_V3_Large(num_classes=1000, scale=0.5):
    multi = 1.0
    bneck_conf = partial(InvertResidualConfig, width_multi=multi)
    adjust_channels = partial(InvertResidualConfig.adjust_channel, multi=multi)
    invert_residual_settings = [
        # input_channel, kernel_size, expand_channel, output_channel, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 160, True, "HS", 2),
        bneck_conf(160, 3, 960, 160, True, "HS", 1),
        bneck_conf(160, 5, 960, 160, True, "HS", 1)]
    last_channel = adjust_channels(1280)
    return MobNetV3(invert_residual_settings, last_channel, num_classes, scale)


if __name__ == "__main__":
    m = MobileNet_V3_Large()
    print(m)
    m.eval()
    # fake_inputs = paddle.randn([1, 3, 224, 224], dtype="float32")
    fake_inputs = paddle.randn([1, 3, 640, 640], dtype="float32")
    out = m(fake_inputs)
    print("最终结果的shape为")
    print(out.shape)
    print("1/4, 1/8, 1/16, 1/32特征shape分别为")
    for i in m.feature_four:
        print(i.shape)
