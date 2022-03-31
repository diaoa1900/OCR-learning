from MobileNetV3scale import MobileNet_V3_Large
from det_mobilenet_v3 import MobileNetV3
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # 升维
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        # 上采样使特征图大小一样从而可以相加
        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest', align_mode=1)
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest', align_mode=1)
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest', align_mode=1)

        # 降维
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p4_conv(out3)
        p2 = self.p4_conv(out2)

        # 上采样使特征图大小一样从而可以concat
        p5 = F.upsample(p5, scale_factor=8, mode='nearest', align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode='nearest', align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode='nearest', align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


if __name__ == "__main__":
    # model_backbone = MobileNetV3()
    model_backbone = MobileNet_V3_Large()
    model_backbone.eval()
    fake_input = paddle.randn([1, 3, 640, 640], dtype='float32')
    m_out = model_backbone(fake_input)
    model_fpn = DBFPN(in_channels=model_backbone.out_channels, out_channels=256)
    print(model_fpn)
    fpn_out = model_fpn(m_out)
    print(fpn_out.shape)
