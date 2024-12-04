import torch
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNet, self).__init__()
        # 第一层 GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二层 GCN
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 全连接层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # 节点特征和边关系
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # GCN 第一层，激活函数 ReLU
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        # GCN 第二层
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        # 图级特征池化（全局平均池化）
        x = global_mean_pool(x, data.batch)  # 适用于批量图输入
        # 全连接层
        x = self.fc(x)
        return x