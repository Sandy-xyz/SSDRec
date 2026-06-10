import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
import dgl
class ViewLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ViewLearner, self).__init__()
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1)
        )

    def build_prob_neighbourhood(self, reg, edge_weight, temperature):
        attention = torch.clamp(edge_weight, 0.01, 0.99)

        relaxed_bernoulli = RelaxedBernoulli(temperature=torch.tensor([temperature]).to(attention.device),
                                             probs=attention)

        weighted_adjacency_matrix = relaxed_bernoulli.rsample()
        eps = 0.0
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)

        return weighted_adjacency_matrix

    def forward(self, g, node_emb, return_info=False):
        # 1. 获取图中所有边的端点
        src, dst = g.edges()

        # 2. 获取边两端节点的嵌入表示
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        # 3. 构造边表示并预测边权重（logits）
        # 将边两端节点的嵌入拼接，作为边的特征表示
        edge_emb = torch.cat([emb_src, emb_dst], dim=1)
        # 通过 MLP 对每条边进行打分，输出边的 logits（未归一化权重）
        edge_logits = self.mlp_edge_model(edge_emb)

        # 4. Gumbel-Sigmoid 重参数化（连续可导采样）
        temperature = 1.0  # The temperature parameter can be adjusted if needed 温度参数：控制采样的平滑程度，越小越接近离散采样
        bias = 0.0001  # Small bias to avoid numerical issues 偏置项：防止数值不稳定（log(0)）

        # 从 (bias, 1 - bias) 区间内采样均匀随机噪声
        # Reparameterization trick
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size(), device=edge_logits.device) + (1 - bias)

        # 将均匀分布噪声映射到 Gumbel 空间
        gate_inputs = torch.log(eps) - torch.log(1 - eps)

        # 将噪声与边 logits 相加，并按温度缩放
        gate_inputs = (gate_inputs + edge_logits) / temperature

        # 通过 sigmoid 得到 (0, 1) 之间的边权重（软掩码）
        edge_weight = torch.sigmoid(gate_inputs).squeeze()

        # 5. 构建加权邻接结构并进行边筛选
        # 根据边权重构建概率邻域（例如保留权重较大的边）
        # 0.9 为阈值或保留比例，用于控制图的稀疏性
        weighted_adjacency_matrix = self.build_prob_neighbourhood(g, edge_weight, 0.9)
        # 根据邻接矩阵中非零位置生成掩码
        mask = (weighted_adjacency_matrix != 0)
        # 仅保留被选中的边
        filtered_src = src[mask]
        filtered_dst = dst[mask]

        # 6. 构建新的图视图
        # 使用筛选后的边构造新的 DGL 图
        adj = dgl.graph((filtered_src, filtered_dst), num_nodes=g.num_nodes())

        if return_info:
            return adj, {
                "src": src.detach().cpu(),
                "dst": dst.detach().cpu(),
                "edge_weight": edge_weight.detach().cpu(),
                "mask": mask.detach().cpu()
            }

        return adj

