import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    # Move weight and other tensors to the device where data resides (assume they are passed from outer context)
    weight = weight.to(mul_L_real.device)

    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight)
    data = -1.0 * torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)

    return torch.stack([real, imag])

class MagConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True):
        super(MagConv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, X_real, X_imag, L_real, L_imag):
        residual_real = X_real
        residual_imag = X_imag

        futures = []
        for i in range(len(L_real)):
            futures.append(torch.jit.fork(process,
                                          L_real[i], L_imag[i],
                                          self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(L_real)):
            result.append(torch.jit.wait(futures[i]))
        result = torch.sum(torch.stack(result), dim=0)

        # real = result[0] + residual_real
        # imag = result[1] + residual_imag
        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias


class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
        if img is None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img


class MagNet(nn.Module):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=2, layer=3, dropout=False):
        super(MagNet, self).__init__()

        activation_func = complex_relu_layer

        chebs = [MagConv(in_c=in_c, out_c=num_filter, K=K)]
        chebs.append(activation_func())

        for i in range(1, layer):
            chebs.append(MagConv(in_c=num_filter, out_c=num_filter, K=K))
            chebs.append(activation_func())

        self.Chebs = nn.Sequential(*chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)
        self.cbam = CBAM(channels=num_filter * last_dim)  # Integrate CBAM here
        self.dropout = dropout

    def forward(self, real, imag, L_real, L_imag):
        for module in self.Chebs:
            if isinstance(module, MagConv):
                real, imag = module(real, imag, L_real, L_imag)
            else:
                real, imag = module(real, imag)

        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))

        x = self.cbam(x)

        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        out = self.channel_attention(x)
        x = x * out

        # Spatial Attention
        out = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        out = self.spatial_attention(out)
        x = x * out

        return x