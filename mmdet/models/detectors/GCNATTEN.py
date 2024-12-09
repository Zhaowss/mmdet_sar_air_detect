import torch
from torch_geometric.nn import GCNConv, global_mean_pool


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GraphNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNet, self).__init__()
        # 第一层 GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二层 GCN
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 全连接层，用于生成节点级嵌入
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # 获取批量数据
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        batch = data.batch  # batch 是一个表示每个节点所属图的张量

        # GCN 第一层，激活函数 ReLU
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        # GCN 第二层
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)

        # 全连接层，用于生成节点的最终嵌入
        x = self.fc(x)  # 对每个节点应用全连接层

        # 现在需要将嵌入的结果按图拆分，保持 [batch_size, num_nodes, embedding_dim]
        # x 的形状是 [batch_size * num_nodes, embedding_dim]，我们要恢复它为 [batch_size, num_nodes, embedding_dim]
        batch_size = int(batch.max()) + 1  # 根据 batch 中的最大值计算 batch_size
        num_nodes = len(batch) // batch_size  # 每个图的节点数量

        # 重新调整形状为 [batch_size, num_nodes, embedding_dim]
        x = x.view(batch_size, num_nodes, -1)

        return x