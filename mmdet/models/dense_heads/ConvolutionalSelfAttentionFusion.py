import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalSelfAttentionFusion(nn.Module):
    def __init__(self, in_channels_visual, in_channels_graph, out_channels, kernel_size=3):
        super(ConvolutionalSelfAttentionFusion, self).__init__()

        # 卷积层，用于生成 Q, K, V
        self.query_conv = nn.Conv2d(in_channels_graph, out_channels, kernel_size, padding=kernel_size // 2)  # 图特征作为查询
        self.key_conv = nn.Conv2d(in_channels_visual, out_channels, kernel_size, padding=kernel_size // 2)  # 视觉特征作为键
        self.value_conv = nn.Conv2d(in_channels_visual, out_channels, kernel_size, padding=kernel_size // 2)  # 视觉特征作为值

        # 输出层，用于生成最终的融合特征
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, feats: torch.Tensor, graph_feature: torch.Tensor) -> torch.Tensor:
        """
        特征融合前向传播.
        Args:
            feats (Tensor): 视觉特征, 形状为 [batch_size, channels, height, width]
            graph_feature (Tensor): 图嵌入特征, 形状为 [batch_size, channels, height, width]
        Returns:
            Tensor: 融合后的特征
        """
        # 计算 Q, K, V
        query = self.query_conv(graph_feature)  # 图特征生成查询Q
        key = self.key_conv(feats)  # 视觉特征生成键K
        value = self.value_conv(feats)  # 视觉特征生成值V

        # 计算自注意力权重（通过 Q 和 K 的相似度）
        attention = torch.matmul(query, key.transpose(2, 3))  # 计算 Q 和 K 的相似度 [batch_size, out_channels, H, W]
        attention = attention / (key.size(-1) ** 0.5)  # 缩放，防止梯度消失

        # 通过 Softmax 获得注意力权重
        attention = F.softmax(attention, dim=-1)

        # 对 V 进行加权求和，得到融合特征
        fused_feature = torch.matmul(attention, value)  # [batch_size, out_channels, H, W]

        # 将融合特征通过输出卷积处理
        fused_feature = self.output_conv(fused_feature)

        return fused_feature



