import torch
import torch.nn.functional as F
from torch import nn

# 论文题目： TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation
# 论文链接：https://arxiv.org/pdf/2107.05274
# 代码改进者：一勺汤

class ScaledDotProductAttention(nn.Module):
    '''自注意力模块'''

    def __init__(self, temperature=512, attn_dropout=0.1):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 计算缩放因子，temperature 的平方根
        self.temperature = temperature ** 0.5
        # 定义 Dropout 层，用于防止过拟合，丢弃概率为 attn_dropout
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        # 获取输入张量 x 的维度信息，分别为批量大小、通道数、高度和宽度
        m_batchsize, d, height, width = x.size()
        # 将输入张量 x 调整形状，将高度和宽度维度合并为一个维度
        q = x.view(m_batchsize, d, -1)
        # 将输入张量 x 调整形状，将高度和宽度维度合并为一个维度
        k = x.view(m_batchsize, d, -1)
        # 对 k 进行转置操作，交换最后两个维度，以便后续进行矩阵乘法
        k = k.permute(0, 2, 1)
        # 将输入张量 x 调整形状，将高度和宽度维度合并为一个维度
        v = x.view(m_batchsize, d, -1)

        # 计算查询矩阵 q 与键矩阵 k 的转置的矩阵乘法，并除以缩放因子 temperature
        attn = torch.matmul(q / self.temperature, k)

        # 如果提供了掩码矩阵 mask
        if mask is not None:
            # 给需要掩码的地方设置一个负无穷，这样在后续的 softmax 操作中，这些位置的概率会趋近于 0
            attn = attn.masked_fill(mask == 0, -1e9)

        # 对注意力分数矩阵 attn 进行 softmax 操作，使其成为概率分布
        # 并使用 Dropout 层随机丢弃一些元素，防止过拟合
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 计算注意力分数矩阵 attn 与值矩阵 v 的矩阵乘法
        output = torch.matmul(attn, v)
        # 将输出张量调整回原来的形状，即恢复高度和宽度维度
        output = output.view(m_batchsize, d, height, width)

        return output


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_ScaledDotProductAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = ScaledDotProductAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_TSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_ScaledDotProductAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":

    # 定义缩放因子
    temperature = 2
    # 生成随机输入张量
    input_tensor = torch.randn(1, 128, 64, 64)
    # 创建 ScaledDotProductAttention 模块实例
    attention_module = ScaledDotProductAttention(128)
    # 调用注意力模块进行前向传播
    output = attention_module(input_tensor)
    # 打印输出张量的形状
    print("Output shape:", output.shape)


# elif m is ScaledDotProductAttention:
#     args = [ch[f]]



# # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# # Ultralytics YOLOv8 object detection model with P3/8 - P5/32 outputs
# # Model docs: https://docs.ultralytics.com/models/yolov8
# # Task docs: https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
#   - [ -1, 1, ScaledDotProductAttention, [ ] ]
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)
#
#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 13], 1, Concat, [1]] # cat head P4
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
#
#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 9], 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
#
#   - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)




# # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# # Ultralytics YOLOv8 object detection model with P3/8 - P5/32 outputs
# # Model docs: https://docs.ultralytics.com/models/yolov8
# # Task docs: https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f_TSA, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f_TSA, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f_TSA, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f_TSA, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)
#
#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 12], 1, Concat, [1]] # cat head P4
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
#
#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 9], 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)


