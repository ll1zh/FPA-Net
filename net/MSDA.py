import torch
import torch.nn as nn
import math

class MutilScaleDualAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        # --------- 三路 depthwise 主体：3x3、~5x5（d=2 的 5x5）、3x3(d=3) ----------
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)                    # 3x3
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim,   # 5x5 with dilation=2
                                      dilation=2)
        self.conv_dilated = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim,   # 3x3(d=3)
                                      dilation=3)

        # --------- 每路先用 1x1 压到 c 通道，最后再拼回去 ----------
        c = math.ceil(dim / 3)     # 取上整，确保 3c >= dim，表达力不丢
        self._c = c
        self.conv1 = nn.Conv2d(dim, c, 1)   # 对应 3x3 路
        self.conv2 = nn.Conv2d(dim, c, 1)   # 对应 “5x5(d=2)” 路
        self.conv3 = nn.Conv2d(dim, c, 1)   # 对应 3x3(d=3) 路

        # --------- 空间聚合：avg/max → 7x7 conv → 3 张分支空间图 ----------
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)

        # --------- 融合后用 1x1 投回 dim，并做门控 ----------
        self.conv = nn.Conv2d(3 * c, dim, 1)

        # --------- 通道注意力（对拼接后的 3c 通道做 SE），保持原逻辑 ----------
        d = max(dim // r, L)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(3 * c, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, 3 * c, 1, 1, bias=False)  # 直接回到 3c，方便按分支拆分
        self.softmax = nn.Softmax(dim=1)                  # 跨分支 softmax（3 路）

    def forward(self, x):
        b, dim, h, w = x.size()

        # 三路 DW 特征
        f1 = self.conv0(x)          # 3x3
        f2 = self.conv_spatial(f1)  # ~5x5(d=2)
        f3 = self.conv_dilated(f1)  # 3x3(d=3)

        # 每路压到 c 通道
        c = self._c
        a1 = self.conv1(f1)         # (b,c,h,w)
        a2 = self.conv2(f2)         # (b,c,h,w)
        a3 = self.conv3(f3)         # (b,c,h,w)

        # 拼接供注意力使用
        attn = torch.cat([a1, a2, a3], dim=1)  # (b,3c,h,w)

        # --- 空间分支权重：avg/max → 7x7 conv 得到三张分支空间图 ---
        avg_attn = torch.mean(attn, dim=1, keepdim=True)                # (b,1,h,w)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)              # (b,1,h,w)
        spa = self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)) # (b,3,h,w)
        w1s, w2s, w3s = spa[:,0:1], spa[:,1:2], spa[:,2:3]              # 每路一张空间图

        # --- 通道分支权重：对 3c 通道做 SE，再按分支拆开并跨分支 softmax ---
        ch = self.global_pool(attn)                                     # (b,3c,1,1)
        z  = self.fc1(ch)                                               # (b,d,1,1)
        a_b = self.fc2(z).reshape(b, 3, c, 1, 1)                        # (b,3,c,1,1)
        a_b = self.softmax(a_b)                                         # 跨 3 路 softmax
        w1c, w2c, w3c = a_b[:,0], a_b[:,1], a_b[:,2]                    # (b,c,1,1)

        # 按（通道 * 空间）对三路加权，并融合
        y1 = a1 * (w1c * w1s)
        y2 = a2 * (w2c * w2s)
        y3 = a3 * (w3c * w3s)
        y  = torch.cat([y1, y2, y3], dim=1)                             # (b,3c,h,w)

        # 门控并输出（与原始 MSDA 一样：x * sigmoid(1x1(...))）
        attn_out = self.conv(y).sigmoid()                               # (b,dim,h,w)
        return x * attn_out

if __name__ == '__main__':

    model = MutilScaleDualAttention(32)
    input_tensor = torch.rand(8, 32, 60, 60)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # 验证输出形状