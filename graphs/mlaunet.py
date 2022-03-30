
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.dau import DAU
import numpy as np


def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def get_attention(attention_type, num_channels=None):
    if attention_type == 'none':
        return None
    else:
        raise Exception('Unknown attention {}'.format(attention_type))


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))

    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()

        self.conv1_1 = conv_block(inplanes, planes, kernel_size=1, padding=0, stride=stride, dilation=dilation,
                                  batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv2 = conv_block(planes, planes, kernel_size=3, padding=1, dilation=dilation, batch_norm=batch_norm,
                                activation='none', padding_mode=padding_mode)

        self.downsample = downsample
        self.stride = stride

        self.activation = get_activation(activation, num_channels=planes)
        self.attention = get_attention(attention_type=attention, num_channels=planes)

    def forward(self, x):
        residual = self.conv1_1(x)

        out = self.activation(self.conv2(self.activation(self.conv1(x))))

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual

        return out


class SpatialAttentionModule(nn.Module):

    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        out = x1 * att_map
        return out


# conv + relu 똑같은거 3번
class ThreeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ThreeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# conv + relu 똑같은거 1번
class OneConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OneConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# conv + relu 똑같은거 1번
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear:
        #
        #   self.up=nn.Upsample(scale_factor=2,mode='bileanear',align_corners=True) #간단 .가중치 학습x
        #   self.conv=DoubleConv(in_channels,out_channels,inchannels//2)
        # else:
        # 2x2 convolution / upsampling 할때마다 채널수가 절반으로 줄어듬
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 각 expanding step 마다 up-conv 된 feature map은 contracting path의 cropped 된 특징맵과 concat
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):  # 1채널(refer),2채널,최종채널
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            #   nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            #   nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttentionNet3(nn.Module):
    def __init__(self):
        super(AttentionNet3, self).__init__()
        self.encoder = DoubleConv(6, 64)
        self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)

        self.att2 = SpatialAttentionModule(128)
        self.att1 = SpatialAttentionModule(64)

        # self.att2 = DAU(128)
        # self.att1 = DAU(64)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)


        self.DRDB = DRDB(in_ch=64)



        self.conv0_2 = DoubleConv(64 * 3, 64)
        self.conv1_2 = DoubleConv(128 * 3, 128)
        self.conv1 = DoubleConv(64 * 2, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),  # 원래 첫번째꺼랑 똑가탔음
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        feat1_1 = self.encoder(x1)  # 6 -> 64
        refer_1 = self.encoder(x2)
        feat3_1 = self.encoder(x3)

        feat1_2 = self.down1(feat1_1)  # 64 -> 128
        refer_2 = self.down1(refer_1)
        feat3_2 = self.down1(feat3_1)

        att1_1 = self.att1(feat1_1, refer_1)
        att1_2 = self.att2(feat1_2, refer_2)
        att3_1 = self.att1(feat3_1, refer_1)
        att3_2 = self.att2(feat3_2, refer_2)


        layer1 = torch.cat([att1_1, refer_1, att3_1], 1)  # 64 * 3
        layer2 = torch.cat([att1_2, refer_2, att3_2], 1)  # 128 *3


        layer1_0 = self.conv0_2(layer1)  # 64*3 -> 64
        layer2_0 = self.conv1_2(layer2)  # 128*3 ->128
        layer1_2 = self.conv1(torch.cat([layer1_0, self.up1(layer2_0)], 1))  # 64 +64 + 64  -> 64


        x2 = self.DRDB(layer1_2)
        x3 = self.DRDB(x2)
        x4 = self.DRDB(x3)

        x5 = torch.cat([x2, x3, x4], dim=1)

        x6 = self.conv0_2(x5)
        x6 = F.relu(x6)

        x7 = x6 + refer_1

        output = self.final(x7)

        return output


class MLAUNet(nn.Module):
    def __init__(self):
        super(MLAUNet, self).__init__()
        self.A = AttentionNet3()

    # self.M = MergingNet()
    def forward(self, x1, x2, x3):
        finalout = self.A(x1, x2, x3)
        return finalout


def test_net():
    model = MLAUNet().cuda()

    # input = torch.from_numpy(np.random.rand(2, 6, 512, 512)).float()
    x_1 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    x_2 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    x_3 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    # print(model)
    output = model(x_1, x_2, x_3)
    print(output.shape)
    # print(output)
    # print('model_parameters')
    print(sum(torch.numel(parameter) for parameter in model.parameters()))
    # print(sum(torch.numel(parameter) for parameter in model.A.alignment_net.parameters()))


if __name__ == '__main__':
    test_net()
