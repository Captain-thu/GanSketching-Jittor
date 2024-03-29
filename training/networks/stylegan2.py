from inspect import CO_VARARGS
import math
import random
import numpy

# import torch
# from torch import nn
# from torch.nn import functional as F
import jittor as jt
import jittor.nn as nn

from jittor import misc

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, input):
        # return input * jt.rsqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)
        return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)


def make_kernel(k):
    k = jt.float32(k)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.kernel = kernel

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def execute(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor,
                        down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        # self.register_buffer("kernel", kernel)
        self.kernel = kernel

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def execute(self, input):
        out = upfirdn2d(input, self.kernel, up=1,
                        down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        # self.register_buffer("kernel", kernel)
        self.kernel = kernel

        self.pad = pad

    def execute(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = jt.randn(out_channel, in_channel,
                               kernel_size, kernel_size)
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = jt.zeros(out_channel)

        else:
            self.bias = None

    def execute(self, input):
        out = nn.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = jt.randn(out_dim, in_dim) / lr_mul

        if bias:
            self.bias = jt.float32(bias_init).expand(out_dim)
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def execute(self, input):
        if self.activation:
            out = (self.weight*self.scale).t()
            out = nn.matmul(input, out)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = nn.matmul(input, (self.weight * self.scale).t()
                            ) + self.bias * self.lr_mul

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(
                pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = jt.randn(1, out_channel, in_channel,
                               kernel_size, kernel_size)

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def conv2d(self, input, weight, groups=1, padding=0, stride=1, dilation=1, out_channels=1, output_padding=0):
        if groups == 1:
            return jt.nn.conv_transpose2d(input, weight, padding=padding, stride=stride, output_padding=output_padding, dilation=dilation)
        else:
            N, C, H, W = input.shape
            i, o, k, _ = weight.shape
            oc = out_channels
            CpG = C // groups  # channels per group

            oh = (H-1) * stride + \
                output_padding - 2*padding + 1 + (k-1)*dilation
            ow = (W-1) * stride + \
                output_padding - 2*padding + 1 + (k-1)*dilation
            out_shape = (N, oc, oh, ow)
            shape = [N, groups, oc//groups, CpG,
                     oh, ow, k, k]
            xx = input.reindex(shape, [
                'i0',
                f'i1*{oc//groups}+i2',
                'i4',
                'i5'
            ])
            ww = weight.reindex(shape, [
                f'i1*{oc//groups}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G": groups, "C": C}
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0',  # Nid
                f'i1*{CpG}+i3',  # Gid
                # Hid+Khid
                f'i4*{stride}-{padding}+i6*{dilation}',
                # Wid+KWid
                f'i5*{stride}-{padding}+i7*{dilation}',
            ])
            return y

    def execute(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = 1 / jt.sqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = self.conv2d(input, weight, padding=0, stride=2,
                              groups=batch, out_channels=batch*self.out_channel)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = nn.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = nn.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = jt.zeros(1)

    def execute(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            # noise = image.new_empty(batch, 1, height, width).normal_()
            noise = jt.normal(mean=0, std=1, size=(batch, 1, height, width))

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = jt.randn(1, channel, size, size)

    def execute(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class WShift(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.w_shift = jt.zeros(1, style_dim)

    def execute(self, input):
        out = input + self.w_shift
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def execute(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = jt.zeros((1, 3, 1, 1))

    def execute(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        w_shift=False,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        if w_shift:
            layers.append(WShift(style_dim))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            setattr(self.noises, f'noise_{layer_idx}', jt.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):

        noises = [jt.randn(1, 1, 2 ** 2, 2 ** 2)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(jt.randn(1, 1, 2 ** i, 2 ** i))

        return noises

    def mean_latent(self, n_latent):
        latent_in = jt.randn(
            n_latent, self.style_dim
        )
        latent = self.style(latent_in).mean(0, keepdims=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def execute(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}")
                         for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation *
                    (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.n_latent - inject_index, 1)

            latent = jt.concat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for idx in range(len(self.to_rgbs)):
            conv1 = self.convs[i-1]
            conv2 = self.convs[i]
            noise1 = noise[i]
            noise2 = noise[i+1]
            to_rgb = self.to_rgbs[idx]
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    # def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 8,
            16: 8,
            32: 8,
            64: 8 * channel_multiplier,
            128: 8 * channel_multiplier,
            256: 8 * channel_multiplier,
            512: 8 * channel_multiplier,
            1024: 4 * channel_multiplier,
        }

        # channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4],
                        activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def execute(self, input):
        print('execute discriminator')
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view((group, -1, self.stddev_feat,
                          channel // self.stddev_feat, height, width))

        var = ((stddev - stddev.mean(dim=0))**2).sum(dim=0) / stddev.shape[0]
        stddev = jt.sqrt(jt.float32(var + 1e-8))
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = jt.concat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, size, out_sz, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))
        out_log_sz = int(math.log(out_sz, 2))
        assert out_log_sz <= log_size and out_log_sz > 2

        in_channel = channels[size]

        for i in range(log_size, out_log_sz, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.patch_conv = ConvLayer(in_channel, 1, 3)

    def execute(self, input):
        out = self.convs(input)
        out = self.patch_conv(out)
        return out
