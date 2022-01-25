import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv import to_2tuple
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import AdaptivePadding

from modules import DeformableTransformerEncoderLayer
import mmcv


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).
    This module uses deconvolution to upsample feature map in the decoder
    of UNet.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out

class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int, optional): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=32,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

class TransformerBlock(nn.Module):
    def __init__(self,
                 channels,
                 num_transformer=2,
                 mlp_ratio=4):
        super(TransformerBlock, self).__init__()


class DeformableUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=32,
                 classes=2,
                 patch_size=16,
                 patch_stride=None,
                 patch_norm=None,
                 num_heads=8,
                 num_points=4,
                 mlp_ratio=4,
                 num_stage=5,
                 trans_per_stage=2,
                 norm_cfg=None):
        super(DeformableUNet, self).__init__()
        self.patch_embedding = PatchEmbed(in_channels=in_channels,
                                          embed_dims=channels,
                                          conv_type='Conv3d',
                                          kernel_size=patch_size,
                                          stride=patch_stride,
                                          dilation=1,
                                          bias=True,
                                          norm_cfg=patch_norm)
        # TODO: add positional embedding
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(num_points):
            encoder_blocks = []
            if index == 0:

