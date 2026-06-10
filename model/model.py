import utility.losses
import utility.tools
import utility.trainer
from .ViewLearner import ViewLearner
from .autocoder import Autoencoder
from dgl.nn.pytorch import GraphConv
import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
import math
from torch import nn
import utility.losses
import utility.tools
import utility.trainer

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPathPreference(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, u_cf, u_meta):
        # u_cf, u_meta: [N, D]
        return self.net(torch.cat([u_cf, u_meta], dim=-1))  # [N, 1]


class GCRec(nn.Module):
    def __init__(self, config, dataset, user_g, item_g, device):
        super(GCRec, self).__init__()
        self._cached_item_weights = None
        self._cached_user_weights = None
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config.reg_lambda)
        self.ssl_lambda = float(self.config.ssl_lambda)
        self.ib_lambda = float(self.config.ib_lambda)
        self.intra_lambda = float(self.config.intra_lambda)  # 同构/同视图约束损失权重
        self.temperature = float(self.config.temperature)
        self.view_learner = ViewLearner(input_dim=self.config.dim, output_dim=self.config.dim)
        self.IB_size = self.config.IB_size
        self.IB_2_size = int(self.config.IB_size/2)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=int(self.config.dim))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=int(self.config.dim))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix  # 归一化后的稀疏邻接矩阵num_nodes * num_nodes
        self.Graph = utility.tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # # 合并重复边并排序索引，同时移动到指定设备
        self.activation = nn.Sigmoid()
        # hete_information
        self.uu_graph = user_g
        self.ii_graph = item_g
        self.user_autoencoder = nn.ModuleList()
        self.user_compressor = nn.Linear(self.config.dim, self.config.IB_size)
        self.item_compressor = nn.Linear(self.config.dim, self.config.IB_size)

        # 为每一种用户 meta-path 子图构建一个自编码器
        for i in range(len(user_g)):
            self.user_autoencoder.append(
                Autoencoder(
                    in_dim=config.in_size,  # 输入特征维度
                    hidden_dim=config.out_size,  # 隐层维度
                    enc_num_layer=config.enc_num_layer,  # 编码器层数
                    dec_num_layer=config.dec_num_layer,  # 解码器层数
                    mask_rate=config.mask_rate,  # 节点/特征掩码比例
                    remask_rate=config.remask_rate,  # 重掩码比例
                    num_remasking=config.num_remasking))  # 重掩码次数
            # ablation
            # self.user_autoencoder.append(GraphConv(config.in_size, config.out_size, bias=False, weight=False,
            #                               allow_zero_in_degree=True))
        self.item_autoencoder = nn.ModuleList()
        for i in range(len(item_g)):
            self.item_autoencoder.append(
                Autoencoder(
                    in_dim=config.in_size,
                    hidden_dim=config.out_size,
                    enc_num_layer=config.enc_num_layer,
                    dec_num_layer=config.dec_num_layer,
                    mask_rate=config.mask_rate,
                    remask_rate=config.remask_rate,
                    num_remasking=config.num_remasking))
            # ablation
            # self.item_autoencoder.append(GraphConv(config.in_size, config.out_size, bias=False, weight=False,
            #                                        allow_zero_in_degree=True))

        self.user_pref = nn.ModuleList([
            MetaPathPreference(int(self.config.dim))
            for _ in range(len(self.uu_graph))
        ])

        self.item_pref = nn.ModuleList([
            MetaPathPreference(int(self.config.dim))
            for _ in range(len(self.ii_graph))
        ])

    def aggregate(self):
        # [user + item, emb_dim] LightGCN
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []

        for layer in range(int(self.config.GCN_layer)):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative, epoch=None):
        # 1. 协同过滤表示（LightGCN）
        user_embeddings, item_embeddings = self.aggregate()
        # hete_emb
        # 2. 异构视图表示学习（Heterogeneous Views）
        hete_user_embedding = []
        hete_item_embedding = []

        # ===== 用户侧 =====
        for i in range(len(self.uu_graph)):
            # denoised_user_emb = self.user_fourier[i](self.user_embedding.weight)
            hete_user_embedding.append(
                self.user_autoencoder[i](
                    self.uu_graph[i],  # 元路径子图
                    self.user_embedding.weight,  # 节点信号
                    is_item=0
                ).flatten(1)
            )

        # ===== 物品侧 =====
        for i in range(len(self.ii_graph)):
            # denoised_item_emb = self.item_fourier[i](self.item_embedding.weight)
            hete_item_embedding.append(
                self.item_autoencoder[i](
                    self.ii_graph[i],
                    self.item_embedding.weight,
                    is_item=1
                ).flatten(1)
            )

        # 3. 信息瓶颈（Information Bottleneck, IB）
        # ib_loss
        # 对多个异构视图的表示进行平均，获得统一的节点表示
        user_node_embs = torch.mean(torch.stack(hete_user_embedding, 0), dim=0)

        # 通过线性压缩器映射到信息瓶颈空间
        user_node_embs = self.user_compressor(user_node_embs)

        # ---------- 变分信息瓶颈：参数拆分 ----------
        # 前一半作为均值 μ
        # KL divergence
        user_mu = user_node_embs[:, :self.IB_2_size]
        # 后一半作为标准差 σ（softplus 保证正值）
        user_std = F.softplus(user_node_embs[:, self.IB_2_size:] - self.IB_2_size, beta=1)

        # ---------- KL 散度（与标准正态分布） ----------
        # KL(q(z|x) || p(z))，用于约束信息压缩
        user_kl_loss = -0.5 * (
                1 + 2 * user_std.log() - user_mu.pow(2) - user_std.pow(2)
        ).sum(1).mean().div(math.log(2))

        # IB 总损失
        ib_loss = self.ib_lambda * user_kl_loss

        # 4. 多视图表示融合
        # ===== MAPM-style Meta-path Preference =====
        user_scores = []
        for i in range(len(hete_user_embedding)):
            s = self.user_pref[i](user_embeddings, hete_user_embedding[i])  # [N, 1]
            user_scores.append(s)

        # [N, M]
        user_scores = torch.cat(user_scores, dim=1)
        user_weights = torch.softmax(user_scores, dim=1)

        # 加权融合所有 meta-path 表示
        user_meta_fused = 0
        for i in range(len(hete_user_embedding)):
            w = user_weights[:, i:i + 1]
            user_meta_fused = user_meta_fused + w * hete_user_embedding[i]

        item_scores = []
        for i in range(len(hete_item_embedding)):
            s = self.item_pref[i](item_embeddings, hete_item_embedding[i])
            item_scores.append(s)

        item_scores = torch.cat(item_scores, dim=1)
        item_weights = torch.softmax(item_scores, dim=1)

        item_meta_fused = 0
        for i in range(len(hete_item_embedding)):
            w = item_weights[:, i:i + 1]
            item_meta_fused = item_meta_fused + w * hete_item_embedding[i]

        # ===== 构造两个 view embedding（用于 DCL）=====
        # 假设你至少有两个 meta-path
        # user_scores: [N, M]，user_weights: softmax 后的权重

        # 取第 0、1 个 meta-path 作为两个视图
        w1 = user_weights[:, 0:1]  # [N, 1]
        w2 = user_weights[:, 1:2]  # [N, 1]

        user_embedding_1 = user_embeddings + w1 * hete_user_embedding[0]
        user_embedding_2 = user_embeddings + w2 * hete_user_embedding[1]

        w1_i = item_weights[:, 0:1]
        w2_i = item_weights[:, 1:2]

        item_embedding_1 = item_embeddings + w1_i * hete_item_embedding[0]
        item_embedding_2 = item_embeddings + w2_i * hete_item_embedding[1]

        # 最终用户 / 物品表示（弱融合，防止异构噪声过大）
        all_user_embeddings = user_embeddings + 1e-2 * user_meta_fused
        all_item_embeddings = item_embeddings + 1e-2 * item_meta_fused

        # 5. 视图内对比学习（Intra-view Contrast）
        # intra-contrast
        user_loss = []
        item_loss = []

        # 用户侧：不同异构视图之间的对比学习
        user_loss.append(
            utility.losses.get_InfoNCE_loss(
                hete_user_embedding[0][user.long()],
                hete_user_embedding[1][user.long()],
                self.temperature))
        # 物品侧：不同异构视图之间的对比学习
        item_loss.append(
            utility.losses.get_InfoNCE_loss(
                hete_item_embedding[0][positive.long()],
                hete_item_embedding[1][positive.long()],
                self.temperature))

        # 视图内对比损失
        user_intra_loss = torch.sum(torch.stack(user_loss))
        item_intra_loss = torch.sum(torch.stack(item_loss))
        intra_loss = self.intra_lambda * (user_intra_loss + item_intra_loss)

        # 6. BPR 推荐损失
        # 取出 batch 中用户、正样本、负样本的最终表示
        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        # 原始（ego）嵌入，用于正则化
        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        # bpr_loss
        bpr_loss = utility.losses.get_bpr_loss(
            user_embedding, pos_embedding, neg_embedding)

        # 7. 正则化损失
        # reg_loss
        reg_loss = utility.losses.get_reg_loss(
            ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        # 8. 视图间对比学习（Inter-view Contrast）
        # CF 表示 + 异构视图表示之间的对比
        user_ssl_loss = utility.losses.get_InfoNCE_loss(
            user_embedding_1[user.long()],
            user_embedding_2[user.long()],
            self.temperature)
        item_ssl_loss = utility.losses.get_InfoNCE_loss(
            item_embedding_1[positive.long()],
            item_embedding_2[positive.long()],
            self.temperature)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        # 9. 总损失
        loss_list = [bpr_loss, reg_loss, ssl_loss, intra_loss, ib_loss]

        return loss_list

    def compute_embeddings(self):
        user_embeddings, item_embeddings = self.aggregate()

        hete_user_embedding = []
        hete_item_embedding = []

        for i in range(len(self.uu_graph)):
            hete_user_embedding.append(
                self.user_autoencoder[i](
                    self.uu_graph[i],
                    self.user_embedding.weight,
                    is_item=0
                ).flatten(1)
            )

        for i in range(len(self.ii_graph)):
            hete_item_embedding.append(
                self.item_autoencoder[i](
                    self.ii_graph[i],
                    self.item_embedding.weight,
                    is_item=1
                ).flatten(1)
            )

        # user preference
        user_scores = torch.cat([
            self.user_pref[i](user_embeddings, hete_user_embedding[i])
            for i in range(len(hete_user_embedding))
        ], dim=1)
        user_weights = torch.softmax(user_scores, dim=1)

        user_meta_fused = 0
        for i in range(len(hete_user_embedding)):
            user_meta_fused += user_weights[:, i:i + 1] * hete_user_embedding[i]

        # item preference
        item_scores = torch.cat([
            self.item_pref[i](item_embeddings, hete_item_embedding[i])
            for i in range(len(hete_item_embedding))
        ], dim=1)
        item_weights = torch.softmax(item_scores, dim=1)

        item_meta_fused = 0
        for i in range(len(hete_item_embedding)):
            item_meta_fused += item_weights[:, i:i + 1] * hete_item_embedding[i]

        all_user_embeddings = user_embeddings + 1e-2 * user_meta_fused
        all_item_embeddings = item_embeddings + 1e-2 * item_meta_fused

        return all_user_embeddings, all_item_embeddings

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.compute_embeddings()

        user_embeddings = all_user_embeddings[user.long()]
        rating = self.activation(
            torch.matmul(user_embeddings, all_item_embeddings.t())
        )
        return rating

    def get_embedding(self):
        all_user_embeddings, all_item_embeddings = self.aggregate()
        return all_user_embeddings, all_item_embeddings

    def get_user_metapath_weights(self):
        """
        Returns:
            Tensor [num_users, num_meta_paths]
        """
        return self._cached_user_weights

    def get_item_embedding_before_denoise(self):
        """
        item embedding before spectral denoising
        """
        item_emb = self.item_embedding.weight.detach()

        return item_emb.cpu()

    def get_item_embedding_after_denoise(self):
        """
        item embedding after spectral denoising
        """
        item_emb = self.item_embedding.weight

        denoised_list = []

        for i in range(len(self.ii_graph)):
            g = self.ii_graph[i]

            denoised = self.item_autoencoder[i].spectral_denoiser(g, item_emb)

            denoised_list.append(denoised)

        denoised_emb = torch.mean(torch.stack(denoised_list), dim=0)

        return denoised_emb.detach().cpu()