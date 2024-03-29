import jittor as jt


class FusedLeakyReLU(jt.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = jt.zeros(channel)
        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def execute(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return jt.nn.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope) * scale
    else:
        return jt.nn.leaky_relu(input, negative_slope) * scale
