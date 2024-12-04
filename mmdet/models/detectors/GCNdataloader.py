from torch_geometric.data import Data, DataLoader
import torch

# 假设 all_corner_tensors 和 graph_features 已经构建完成
def create_graph_data(all_corner_tensors, graph_features):
    graph_data_list = []
    for i in range(all_corner_tensors.shape[0]):
        # 节点特征 (x, y)
        node_features = all_corner_tensors[i]  # [N, 2]
        # 邻接矩阵转换为边索引和权重
        edge_index = torch.nonzero(graph_features[i] > 0, as_tuple=False).t()  # [2, M]
        edge_weight = graph_features[i][edge_index[0], edge_index[1]]  # [M]

        # 创建 PyG 数据对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        graph_data_list.append(graph_data)

    return graph_data_list