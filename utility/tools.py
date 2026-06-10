import os
import torch
import random
import numpy as np
import scipy.sparse as sp


def set_seed(seed):

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def read_configuration(filename, model):
    if not os.path.exists(filename):
        print("\tThe path does not have a configuration file for " + model + ".")
        raise IOError
    else:
        with open(filename, "r") as f:
            config = dict()
            line = f.readline()
            while line is not None and line != "":
                try:
                    name, value = line.strip().split("=")
                    config[name.strip()] = value.strip()
                except ValueError:
                    print("\tConfiguration file format error.")
                line = f.readline()
        return config


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def mini_batch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 1024)

    # 如果只传入了一个 tensor
    if len(tensors) == 1:
        tensor = tensors[0]
        # 按 batch_size 对该 tensor 进行切分
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]  # 每次返回一个 batch

    # 如果传入了多个 tensor（如 users, pos_items, neg_items）
    else:
        # 默认认为所有 tensor 的第 0 维长度一致
        for i in range(0, len(tensors[0]), batch_size):
            # 对每个 tensor 同步切片，保证 batch 内样本一一对应
            yield tuple(x[i:i + batch_size] for x in tensors)


def create_adj_mat(inter_graph, aug_type, ssl_rate):
    graph_shape = inter_graph.get_shape()
    node_number = graph_shape[0] + graph_shape[1]
    user_index, item_index = inter_graph.nonzero()

    if aug_type == 'nd':
        raise NotImplementedError("The method does not implemented.")
    elif aug_type in ['ed', 'rw']:
        edge_number = inter_graph.count_nonzero()

        keep_index = random.sample(range(edge_number), k=int((1 - ssl_rate) * edge_number))
        user_index = np.array(user_index)[keep_index]
        item_index = np.array(item_index)[keep_index]
        ratings = np.ones_like(user_index, dtype=np.float32)
        new_graph = sp.csr_matrix((ratings, (user_index, item_index + graph_shape[0])), shape=(node_number, node_number))
    adjacency_matrix = new_graph + new_graph.T

    row_sum = np.array(adjacency_matrix.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    degree_matrix = sp.diags(d_inv)

    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()

    return norm_adjacency

def convert_sp_mat_to_sp_tensor(sp_mat):
    """
    将 SciPy 的稀疏矩阵转换为 PyTorch 的稀疏张量（sparse tensor）

        coo.row: x in user-item graph 非零元素的行索引（在用户-物品图中对应 x 轴，如用户或节点）
        coo.col: y in user-item graph 非零元素的列索引（在用户-物品图中对应 y 轴，如物品或节点）
        coo.data: [value(x,y)] 每个 (row, col) 位置对应的非零取值
    """
    # 将稀疏矩阵转换为 COO 格式，并统一转为 float32
    coo = sp_mat.tocoo().astype(np.float32)
    # 行索引，转换为 torch 的 long 类型（索引必须是整型）
    row = torch.Tensor(coo.row).long()

    # 列索引，转换为 torch 的 long 类型
    col = torch.Tensor(coo.col).long()

    # 将行索引和列索引堆叠成 shape = (2, nnz) 的索引张量
    # 第一行是 row，第二行是 col
    index = torch.stack([row, col])

    # 非零元素的取值，对应每条边/连接的权重
    value = torch.FloatTensor(coo.data)

    # 根据 index、value 和矩阵原始形状构造 PyTorch 稀疏张量
    # sp_tensor 的大小为 (num_rows, num_cols)
    # from a sparse matrix to a sparse float tensor
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    return sp_tensor
