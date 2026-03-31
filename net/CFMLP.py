import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, nc, expand=2, scale=0.02):
        super(MLP, self).__init__()
        self.scale = scale

        # 复数权重参数 (核心创新)
        self.r_weight = nn.Parameter(self.scale * torch.randn(nc, nc))
        self.i_weight = nn.Parameter(self.scale * torch.randn(nc, nc))
        self.r_bias = nn.Parameter(self.scale * torch.randn(nc))
        self.i_bias = nn.Parameter(self.scale * torch.randn(nc))

        # 原始光照增强路径 (保持核心功能)
        self.lighting_path = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 空间域 -> 频域
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        # 2. 核心光照增强
        mag_enhanced = self.lighting_path(mag)

        # 3. 复数域处理 (创新点)
        # 将幅度视为"实部"，相位视为"虚部"
        real = mag_enhanced * torch.cos(pha)
        imag = mag_enhanced * torch.sin(pha)

        # 应用复数权重 (核心创新)
        # [B, C, H, W//2+1] -> [B, H, W//2+1, C]
        real = real.permute(0, 2, 3, 1)
        imag = imag.permute(0, 2, 3, 1)

        # 复数线性变换
        o_real = (
                torch.einsum('bhwc,cd->bhwd', real, self.r_weight) -
                torch.einsum('bhwc,cd->bhwd', imag, self.i_weight) +
                self.r_bias
        )

        o_imag = (
                torch.einsum('bhwc,cd->bhwd', imag, self.r_weight) +
                torch.einsum('bhwc,cd->bhwd', real, self.i_weight) +
                self.i_bias
        )

        # 恢复形状 [B, C, H, W//2+1]
        o_real = o_real.permute(0, 3, 1, 2)
        o_imag = o_imag.permute(0, 3, 1, 2)

        # 计算变换后的幅度谱
        mag_out = torch.sqrt(o_real ** 2 + o_imag ** 2)

        # 4. 重组频谱 (保持原始相位)
        real_out = mag_out * torch.cos(pha)
        imag_out = mag_out * torch.sin(pha)
        x_out = torch.complex(real_out, imag_out)

        # 5. 频域 -> 空间域
        return torch.fft.irfft2(x_out, s=(H, W), norm='backward')
class CFMLP(nn.Module):
    def __init__(self, channels):
        super(CFMLP, self).__init__()  # 先调用父类初始化
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.norm = nn.LayerNorm(channels)
        self.freq = MLP(channels)  # 使用 FreMLP 模块

    def forward(self, inp):
        B, C, H, W = inp.shape
        x_permuted = inp.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm(x_permuted)
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 2. 频域处理
        x_freq = self.freq(x_norm)

        # 3. 残差连接与缩放
        x = inp + x_freq * self.gamma
        return x


# if __name__ == '__main__':
#
#     model = CFMLP(32)
#     input_tensor = torch.rand(1, 32, 60, 60)
#     output_tensor = model(input_tensor)
#     print(output_tensor.shape)  # 验证输出形状