from torch_geometric.data import Data, DataLoader
import torch

# 假设 all_corner_tensors 和 graph_features 已经构建完成
def create_graph_data(all_corner_tensors, graph_features, top_k=5):
    graph_data_list = []

    # 计算每个batch的图数据
    for i in range(all_corner_tensors.shape[0]):
        # 节点特征 (x, y)，[num_nodes, 2]
        node_features = all_corner_tensors[i]

        # 获取当前图像的相似度矩阵
        similarity_matrix = graph_features[i]  # [num_nodes, num_nodes]

        # 对每个节点，选择最相似的top_k个节点
        edge_index = []
        edge_weight = []

        for j in range(similarity_matrix.shape[0]):
            # 获取当前节点的相似度向量（与其他节点的相似度）
            similarities = similarity_matrix[j]
            # 找到top_k个相似度最大的节点（排除自己）
            top_k_similarities, top_k_indices = torch.topk(similarities, top_k+1, largest=True)  # top_k+1是因为包含自己
            top_k_indices = top_k_indices[1:]  # 排除自己
            top_k_similarities = top_k_similarities[1:]  # 排除自己

            for node_idx, similarity in zip(top_k_indices, top_k_similarities):
                edge_index.append([j, node_idx])  # 存储边的起始节点和结束节点
                edge_weight.append(similarity)  # 使用相似度作为边的权重

        edge_index = torch.tensor(edge_index).t().cuda()  # 转置为 [2, M]
        edge_weight = torch.tensor(edge_weight).cuda()  # [M]

        # 创建 PyG 数据对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        graph_data_list.append(graph_data)

    return graph_data_list