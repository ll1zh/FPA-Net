import torch
import torch.nn as nn
from net.transformer_utils import *  # 假设原有LayerNorm等
from einops import rearrange

class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)

        self.pos_embed = nn.Parameter(torch.randn(1, dim // num_heads, 1, 1))
        self.pos_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        head = self.num_heads
        d_head = c // head

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.temperature1  # b head d seq @ b head seq d = b head d d
        attn = F.softmax(attn, dim=-1)
        out_attn = (attn @ v)  # b head d d @ b head d seq = b head d seq
        out_attn = rearrange(out_attn, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        pos_x = self.pos_conv(x)
        pos_embed_exp = self.pos_embed.repeat(b, head, h, w)
        pos_x = pos_x + pos_embed_exp

        pos_q = pos_x.view(b, head, d_head, h * w)
        pos_q = F.normalize(pos_q, dim=-1)

        pos_attn = (pos_q @ k.permute(0, 1, 3, 2)) * self.temperature2
        pos_attn = F.softmax(pos_attn, dim=-1)
        pos_out = (pos_attn @ v)
        pos_out = pos_out.view(b, head, d_head, h, w).view(b, c, h, w)

        # 融合
        out = out_attn * self.alpha + pos_out * (1-self.alpha)

        out = self.project_out(out)
        return out


class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention (替换ffn为PosCAB)
class HV_PACA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)  # 使用PosCAB

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_PACA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)  # 使用PosCAB

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


if __name__ == '__main__':
    model = CAB(32, 4, True)
    x = torch.rand(1, 32, 256, 256)
    y = torch.rand(1, 32, 256, 256)
    output_tensor = model(x, y)
    print(output_tensor.shape)  # 验证输出形状