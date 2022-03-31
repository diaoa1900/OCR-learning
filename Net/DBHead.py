import paddle
from MobileNetV3scale import MobileNet_V3_Large
from DBFPN import DBFPN


class Head(paddle.nn.Layer):
    def __init__(self, in_channels):
        super(Head, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.bn1 = paddle.nn.BatchNorm2D(in_channels // 4)
        self.activation1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2DTranspose(in_channels // 4, in_channels // 4, kernel_size=2, stride=2)
        self.bn2 = paddle.nn.BatchNorm2D(in_channels // 4)
        self.activation2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2DTranspose(in_channels // 4, 1, kernel_size=2, stride=2)
        self.activation3 = paddle.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        return x


class DBHead(paddle.nn.Layer):
    def __init__(self, in_channels, k=50):
        super(DBHead, self).__init__()
        self.binarize = Head(in_channels)
        self.threshold = Head(in_channels)
        self.k = k

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if self.training:
            threshold_maps = self.threshold(x)
            binary_maps = paddle.reciprocal(1 + paddle.exp(-self.k * (shrink_maps-threshold_maps)))
            y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
            return {'maps': y}
        return {'maps': shrink_maps}


if __name__ == '__main__':
    fake_input = paddle.randn([1, 3, 640, 640], dtype="float32")
    backbone = MobileNet_V3_Large()
    m_out = backbone(fake_input)
    fpn = DBFPN(in_channels=backbone.out_channels, out_channels=256)
    fpn_out = fpn(m_out)
    head = DBHead(in_channels=256)
    print(head)
    head_out = head(fpn_out)
    print(head_out['maps'].shape)
    print(head_out)
