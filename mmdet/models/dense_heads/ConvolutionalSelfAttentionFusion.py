import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalSelfAttentionFusion(nn.Module):
    def __init__(self, in_channels_visual, in_channels_graph, out_channels, kernel_size=1):
        super(ConvolutionalSelfAttentionFusion, self).__init__()

        # 第一分支: 图特征作为查询，视觉特征作为键和值
        self.query_conv_graph = nn.Conv2d(in_channels_graph, out_channels, kernel_size, padding=kernel_size // 2)
        self.key_conv_visual = nn.Conv2d(in_channels_visual, out_channels, kernel_size, padding=kernel_size // 2)
        self.value_conv_visual = nn.Conv2d(in_channels_visual, out_channels, kernel_size, padding=kernel_size // 2)

        # 第二分支: 视觉特征作为查询，图特征作为键和值
        self.query_conv_visual = nn.Conv2d(in_channels_visual, out_channels, kernel_size, padding=kernel_size // 2)
        self.key_conv_graph = nn.Conv2d(in_channels_graph, out_channels, kernel_size, padding=kernel_size // 2)
        self.value_conv_graph = nn.Conv2d(in_channels_graph, out_channels, kernel_size, padding=kernel_size // 2)

        # 输出层，用于生成最终的融合特征
        self.output_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, feats: torch.Tensor, graph_feature: torch.Tensor) -> torch.Tensor:
        """
        特征融合前向传播.
        Args:
            feats (Tensor): 视觉特征, 形状为 [batch_size, channels, height, width]
            graph_feature (Tensor): 图嵌入特征, 形状为 [batch_size, channels, height, width]
        Returns:
            Tensor: 融合后的特征
        """
        # 第一分支: 图特征作为查询，视觉特征作为键和值
        query_graph = self.query_conv_graph(graph_feature)
        key_visual = self.key_conv_visual(feats)
        value_visual = self.value_conv_visual(feats)

        attention_graph = torch.matmul(query_graph, key_visual.transpose(2, 3))  # [batch_size, out_channels, H, W]
        attention_graph = attention_graph / (key_visual.size(-1) ** 0.5)  # 缩放
        attention_graph = F.softmax(attention_graph, dim=-1)
        fused_graph = torch.matmul(attention_graph, value_visual)

        # 第二分支: 视觉特征作为查询，图特征作为键和值
        query_visual = self.query_conv_visual(feats)
        key_graph = self.key_conv_graph(graph_feature)
        value_graph = self.value_conv_graph(graph_feature)

        attention_visual = torch.matmul(query_visual, key_graph.transpose(2, 3))  # [batch_size, out_channels, H, W]
        attention_visual = attention_visual / (key_graph.size(-1) ** 0.5)  # 缩放
        attention_visual = F.softmax(attention_visual, dim=-1)
        fused_visual = torch.matmul(attention_visual, value_graph)

        # 融合两分支特征
        concatenated_feature = torch.cat([fused_graph, fused_visual], dim=1)  # 在通道维度上拼接

        # 将融合特征通过输出卷积处理
        fused_feature = self.output_conv(concatenated_feature)

        return fused_feature
