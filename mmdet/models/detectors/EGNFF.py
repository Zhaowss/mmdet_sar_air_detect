import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# 定义 EnhancedGraphNetWithFeatureFusion 模型
class EnhancedGraphNetWithFeatureFusion(nn.Module):
    def __init__(self,  hidden_dim, num_fpn_layers):
        super(EnhancedGraphNetWithFeatureFusion, self).__init__()

        # 通道加权层（用于浅层GCN特征）
        self.channel_weight_layer = nn.Linear(hidden_dim, 1)

        # 全局加权层（用于深层GCN特征）
        self.global_weight_layer = nn.Linear(hidden_dim, num_fpn_layers)


    def forward(self, gcn_layers, fpn_outputs):
        """
        Args:
            gcn_layers: GCN的不同层级特征的列表，假设包含浅层和深层GCN特征
            visual_features: 来自视觉的特征图 [batch_size, channels, H, W]
            fpn_outputs: 特征金字塔输出，tuple，每个元素为特征图张量 [batch_size, channels, H, W]

        Returns:
            enhanced_fpn_outputs: 加权后的特征金字塔输出，tuple
            final_embedding: 融合后的最终嵌入
        """
        # 假设gcn_layers是一个包含不同层特征的列表
        # gcn_layers[0] -> 浅层GCN特征
        # gcn_layers[1] -> 中层GCN特征
        # gcn_layers[2] -> 深层GCN特征
        gcn_layer_1, gcn_layer_2, gcn_layer_3 = gcn_layers

        # ======= 浅层GCN特征加权（通道加权） =======
        x1_out = self.channel_weight_layer(gcn_layer_1)  # 计算通道权重
        channel_weights = torch.sigmoid(x1_out)  # 通道加权系数 [num_nodes, 1]

        # ======= 深层GCN特征加权（全局加权） =======
        global_weights = torch.sigmoid(self.global_weight_layer(gcn_layer_3.mean(dim=0)))  # 全局加权 [num_fpn_layers]

        # ======= 加权特征金字塔输出 =======
        enhanced_fpn_outputs = []
        for i, fpn_feature in enumerate(fpn_outputs):
            # 获取当前特征图形状
            batch_size, channels, H, W = fpn_feature.shape

            # 通道加权（浅层GCN提供）
            channel_weights_broadcast = channel_weights.view(batch_size, channels, 1, 1)
            fpn_feature = fpn_feature * channel_weights_broadcast  # 通道加权

            # 全局加权（深层GCN提供）
            global_weight = global_weights[i].view(1, 1, 1, 1)
            fpn_feature = fpn_feature * global_weight  # 全局加权

            enhanced_fpn_outputs.append(fpn_feature)
        return tuple(enhanced_fpn_outputs)

