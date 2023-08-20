import pickle
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import numpy
import torch_geometric.transforms as T
from torch_geometric.data import Data


# %%
def select_subgraph(graph, label_select):
    '''
    :param graph: the object of graph data processing
    :param label_select: The number of labels of the subgraph generated
    :return: subgraph data; mapping of subgraph node ID to original graph node ID
    '''
    num_nodes = graph.num_nodes
    mask = graph.y.squeeze() < label_select
    graph.y = graph.y.squeeze()[mask]
    graph.x = graph.x[mask]
    graph.node_year = graph.node_year[mask]

    index_map = torch.zeros(num_nodes, dtype=torch.long)
    index_map[mask] = torch.arange(mask.sum())

    edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
    row, col = graph.edge_index[:, edge_mask]
    graph.edge_index = torch.stack([index_map[row], index_map[col]], dim=0)
    graph.num_nodes = mask.sum().item()
    sub_graph = Data(num_nodes=int(graph.num_nodes), x=graph.x, edge_index=graph.edge_index, y=graph.y)

    #
    connected_nodes = torch.unique(sub_graph.edge_index)
    mask_1 = torch.isin(torch.arange(sub_graph.num_nodes), connected_nodes)  # torch.isin(a,b)可以判断b中元素是否在a里面
    sub_graph.y = sub_graph.y[mask_1]
    sub_graph.x = sub_graph.x[mask_1]

    sub_graph_index_map = torch.zeros(sub_graph.num_nodes, dtype=torch.long)
    sub_graph_index_map[mask_1] = torch.arange(mask_1.sum())
    sub_edge_mask = mask_1[sub_graph.edge_index[0]] & mask_1[sub_graph.edge_index[1]]
    row, col = sub_graph.edge_index[:, sub_edge_mask]
    sub_graph.edge_index = torch.stack([sub_graph_index_map[row], sub_graph_index_map[col]], dim=0)
    sub_graph.num_nodes = int(mask_1.sum().item())

    mask_2 = torch.clone(mask)
    mask_2[mask] = mask_1
    subgraph_indices = torch.where(mask_2)[0]

    node_mapping = dict(zip(torch.arange(mask_1.sum()).tolist(), subgraph_indices.tolist()))

    # %% split data into training set and test set
    data = sub_graph
    split = T.RandomNodeSplit(num_test=0.2, num_val=0)
    data = split(data)
    return data, node_mapping


def train_test(data, mask):
    """
    generating training and test sets
    """
    index_map = torch.zeros(data.num_nodes, dtype=torch.long)
    index_map[mask] = torch.arange(mask.sum())

    # 更新边的源和目标索引
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    row, col = data.edge_index[:, edge_mask]
    edge_index = torch.stack([index_map[row], index_map[col]], dim=0)
    num_nodes = mask.sum().item()
    # 创建新的子图
    sub_data = Data(num_nodes=int(num_nodes), x=data.x[mask], edge_index=edge_index,
                    y=data.y[mask])
    '''
    # 处理孤立节点
    connected_nodes = torch.unique(sub_data.edge_index)
    mask_1 = torch.isin(torch.arange(sub_data.num_nodes),
                        connected_nodes)  # torch.isin(a,b)可以判断b中元素是否在a里面
    sub_data.y = sub_data.y[mask_1]
    sub_data.x = sub_data.x[mask_1]

    sub_data_index_map = torch.zeros(sub_data.num_nodes, dtype=torch.long)
    sub_data_index_map[mask_1] = torch.arange(mask_1.sum())
    sub_edge_mask = mask_1[sub_data.edge_index[0]] & mask_1[sub_data.edge_index[1]]
    row1, col1 = sub_data.edge_index[:, sub_edge_mask]
    sub_data.edge_index = torch.stack([sub_data_index_map[row1], sub_data_index_map[col1]], dim=0)
    sub_data.num_nodes = int(mask_1.sum().item())
    '''
    return sub_data


if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    graph = dataset[0]
    select = int(input('selcet_label='))
    data, node_mapping = select_subgraph(graph, select)
    # print(data)
    # print(train_test(data, data.train_mask))
    # print(train_test(data, data.test_mask))
    with open('./data/graph_data+node_mapping(select_label_{}).pkl'.format(select), 'wb') as f:
        pickle.dump(data, f)
    pickle.dump(node_mapping, f)
