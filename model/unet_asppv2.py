import spconv
from spconv.modules import SparseModule
import torch
from torch import nn

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, ,kernel_size =3, padding = 1, stride= 1, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride =stride,  bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding,stride =stride, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class ASPP(SparseModule):
    def __init__(self,in_channels, out_channels, norm_fn, indice_key=6, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = ResidualBlock(inplanes, planes, norm_fn, padding=rate[0], dilation=rate[0], indice_key='bb_subm{}'.format(indice_key))
        
        

        self.aspp_block2 = ResidualBlock(inplanes, planes, norm_fn, padding=rate[1], dilation=rate[1], indice_key='bb_subm{}'.format(indice_key))
        
      
        self.aspp_block3 = ResidualBlock(inplanes, planes, norm_fn, padding=rate[2], dilation=rate[2], indice_key='bb_subm{}'.format(indice_key))
        
       
        self.conv_1x1 = spconv.SparseConv3d(len(rate) * out_channels, out_channels, 1, indice_key=indice_key)


    def forward(self, input):

        x1 = self.aspp_block1(input)
        x2 = self.aspp_block2(input)
        x3 = self.aspp_block3(input)
        x3.features = torch.cat((x1.features, x2.features,x3.features), dim=1)
        out = self.conv_1x1(x3)
        return out



class Unet_aspp(SparseModule):

    def __init__(self, nPlanes, norm_fn, block_reps):
        super().__init__()
        
        self.block0 = self._make_layers(nPlanes[0], nPlanes[0], block_reps, norm_fn, indice_key=0)
        
        self.conv1 = spconv.SparseSequential(
            norm_fn(nPlanes[0]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )
        self.block1 = self._make_layers(nPlanes[1], nPlanes[1], block_reps, norm_fn, indice_key=1)

        self.conv2 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[1], nPlanes[2], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )
        self.block2 = self._make_layers(nPlanes[2], nPlanes[2], block_reps, norm_fn, indice_key=2)

        self.conv3 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[2], nPlanes[3], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )
        self.block3 = self._make_layers(nPlanes[3], nPlanes[3], block_reps, norm_fn, indice_key=3)


        self.conv4 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[3], nPlanes[4], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )
        self.block4 = self._make_layers(nPlanes[4], nPlanes[4], block_reps, norm_fn, indice_key=4)

        self.conv5 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[4], nPlanes[5], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )
        self.block5 = self._make_layers(nPlanes[5], nPlanes[5], block_reps, norm_fn, indice_key=5)

        self.conv6 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[5], nPlanes[6], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )
        self.aspp = ASPP(nPlanes[6], nPlanes[6],  norm_fn, indice_key=6)
        # self.block6 = self._make_layers(nPlanes[6], nPlanes[6], block_reps, norm_fn, indice_key=6)

        self.deconv6 = spconv.SparseSequential(
            norm_fn(nPlanes[6]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )

        self.deblock5 = self._make_layers(nPlanes[5] * 2, nPlanes[5], block_reps, norm_fn, indice_key=5)
        self.deconv5 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[5], nPlanes[4], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )

        self.deblock4 = self._make_layers(nPlanes[4] * 2, nPlanes[4], block_reps, norm_fn, indice_key=4)
        self.deconv4 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[4], nPlanes[3], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )

        self.deblock3 = self._make_layers(nPlanes[3] * 2, nPlanes[3], block_reps, norm_fn, indice_key=3)
        self.deconv3 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[3], nPlanes[2], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )

        self.deblock2 = self._make_layers(nPlanes[2] * 2, nPlanes[2], block_reps, norm_fn, indice_key=2)
        self.deconv2 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[2], nPlanes[1], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )

        self.deblock1 = self._make_layers(nPlanes[1] * 2, nPlanes[1], block_reps, norm_fn, indice_key=1)
        self.deconv1 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )

        self.deblock0 = self._make_layers(nPlanes[0] * 2, nPlanes[0], block_reps, norm_fn, indice_key=0)

        
    def _make_layers(self, inplanes, planes, block_reps, norm_fn, indice_key=0):
        blocks = [ResidualBlock(inplanes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key))]
        for i in range(block_reps - 1):
            blocks.append(ResidualBlock(planes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key)))
        return spconv.SparseSequential(*blocks)

    def forward(self, x):
        out0 = self.block0(x)
        
        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        out5 = self.conv5(out4)
        out5 = self.block5(out5)

        out6 = self.conv6(out5)
        out6 = self.aspp(out6)

        d_out5 = self.deconv6(out6)
        d_out5.features = torch.cat((d_out5.features, out5.features), dim=1)
        d_out5 = self.deblock5(d_out5)

        d_out4 = self.deconv5(d_out5)
        d_out4.features = torch.cat((d_out4.features, out4.features), dim=1)
        d_out4 = self.deblock4(d_out4)

        d_out3 = self.deconv4(d_out4)
        d_out3.features = torch.cat((d_out3.features, out3.features), dim=1)
        d_out3 = self.deblock3(d_out3)

        d_out2 = self.deconv3(d_out3)
        d_out2.features = torch.cat((d_out2.features, out2.features), dim=1)
        d_out2 = self.deblock2(d_out2)

        d_out1 = self.deconv2(d_out2)
        d_out1.features = torch.cat((d_out1.features, out1.features), dim=1)
        d_out1 = self.deblock1(d_out1)

        d_out0 = self.deconv1(d_out1)
        d_out0.features = torch.cat((d_out0.features, out0.features), dim=1)
        d_out0 = self.deblock0(d_out0)

        return d_out0

