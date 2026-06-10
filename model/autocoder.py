from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch
import dgl


# ======================
# Spectral Denoiser
# ======================
class GraphSpectralDenoiser(nn.Module):
    def __init__(self, alpha=0.1, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, g, x):
        with g.local_scope():
            deg = g.in_degrees().float().clamp(min=1)
            norm = (1.0 / deg).unsqueeze(1)

            g.ndata['h'] = x
            g.update_all(
                message_func=dgl.function.copy_u('h', 'm'),
                reduce_func=dgl.function.sum('m', 'h_neigh')
            )
            Ax = g.ndata['h_neigh']

            # X - α (X - D^{-1} A X)
            x_smooth = x - self.alpha * (x - norm * Ax)

        return x_smooth


class AdaptiveSpectralDenoiser(nn.Module):
    def __init__(self, in_dim, alpha=0.1):
        """
        Adaptive Spectral Denoiser
        用 attention 学习 A~，再做 Laplacian smoothing

        参数：
        in_dim: 特征维度
        alpha: 平滑系数
        """
        super().__init__()

        # 平滑系数，设为可学习参数
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # attention 打分函数：输入拼接后的 [z_i || z_j] → 输出 e_ij
        self.attn_fc = nn.Linear(in_dim * 2, 1, bias=False)

        # 可选特征变换（类似 GAT 中的线性映射）
        self.fc = nn.Linear(in_dim, in_dim, bias=False)

        # 非线性激活（GAT 常用）
        self.leaky_relu = nn.LeakyReLU(0.2)

    def edge_attention(self, edges):
        """
        边级别计算 attention score（未归一化）

        对每条边 (i → j)：
        输入：源节点特征 z_i 和目标节点特征 z_j
        输出：边权重 e_ij
        """
        # 拼接源节点和目标节点特征
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [E, 2d]
        # 线性变换 + 激活，得到 attention score
        e = self.leaky_relu(self.attn_fc(z2))  # [E, 1]

        return {'e': e}  # 存入 g.edata['e']

    def forward(self, g, x):
        """
        前向传播

        输入：
        g: DGL 图
        x: 节点特征 [N, d]

        输出：
        x_smooth: 去噪后的节点表示
        """
        with g.local_scope():  # 防止污染原图数据

            # Step1️⃣：特征线性变换（类似 GAT）
            z = self.fc(x)  # [N, d]
            g.ndata['z'] = z  # 存入节点数据

            # Step2️⃣：计算每条边的 attention score e_ij
            g.apply_edges(self.edge_attention)
            # 此时：g.edata['e'] = e_ij

            # Step3️⃣：对每个节点的入边做 softmax 归一化
            # 得到 attention 权重 α_ij
            a = dgl.nn.functional.edge_softmax(g, g.edata['e'])
            g.edata['a'] = a  # 保存归一化后的权重

            # Step4️⃣：消息传递（高效内置函数）
            # u_mul_e: 源节点特征 * 边权重
            # sum: 对邻居求和
            g.update_all(
                dgl.function.u_mul_e('z', 'a', 'm'),  # m_ij = α_ij * z_i
                dgl.function.sum('m', 'h_neigh')  # h_j = Σ α_ij z_i
            )

            # 得到 A~X（自适应邻接矩阵作用结果）
            Ax_tilde = g.ndata['h_neigh']  # [N, d]

            # Step5️⃣：谱域去噪（核心公式）
            # X' = X - α (X - A~X)
            alpha = torch.sigmoid(self.alpha)  # 限制在 (0,1)，更稳定

            # 等价形式：x_smooth = (1 - α)X + α A~X
            x_smooth = x - alpha * (x - Ax_tilde)

        return x_smooth


# ======================
# Autoencoder
# ======================
class Autoencoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        enc_num_layer,
        dec_num_layer,
        mask_rate,
        remask_rate,
        num_remasking
    ):
        super(Autoencoder, self).__init__()

        # ========= Projection =========
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        # ========= Encoder =========
        self.encoder = nn.ModuleList()
        for _ in range(enc_num_layer):
            self.encoder.append(
                GraphConv(
                    hidden_dim,
                    hidden_dim,
                    weight=False,   # 保持你的propagation
                    bias=False,
                    allow_zero_in_degree=True
                )
            )

        # ========= Decoder =========
        self.decoder = nn.ModuleList()
        for _ in range(dec_num_layer):
            self.decoder.append(
                GraphConv(
                    hidden_dim,
                    hidden_dim,
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True
                )
            )

        # ========= Mask =========
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))

        # ========= Output =========
        self.decoder_to_contrastive = nn.Linear(hidden_dim, in_dim, bias=False)

        self.reset_parameters()

        # ========= Spectral Denoising =========
        # self.spectral_denoiser = GraphSpectralDenoiser(learnable=True)
        self.spectral_denoiser = AdaptiveSpectralDenoiser(in_dim=hidden_dim)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.input_proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.decoder_to_contrastive.weight, gain=1.414)

    # ======================
    # Forward
    # ======================
    def forward(self, g, x, drop_g1=None, is_item=1):

        # ===== 1. Mask =====
        pre_use_g, mask_x, _ = self.encoding_mask_noise(
            g, x, self.mask_rate
        )

        use_g = drop_g1 if drop_g1 is not None else g

        # ===== 2. Projection（关键！）=====
        h = self.input_proj(mask_x)

        # ===== 3. Propagation =====
        all_embeddings = []
        for layer in self.encoder:
            h = layer(use_g, h)
            all_embeddings.append(h)

        enc_rep = torch.stack(all_embeddings, dim=1).mean(dim=1)

        # ===== 4. Spectral Denoising =====
        if is_item:
            enc_rep = self.spectral_denoiser(use_g, enc_rep)

        # ===== 5. Decoder（remask）=====
        final_recon = []

        for _ in range(self.num_remasking):
            rep = enc_rep.clone()

            rep, _, _ = self.random_remask(
                pre_use_g, rep, self.remask_rate
            )

            h = rep
            all_dec = []

            for layer in self.decoder:
                h = layer(pre_use_g, h)
                all_dec.append(h)

            dec_rep = torch.stack(all_dec, dim=1).mean(dim=1)
            final_recon.append(dec_rep)

        final_recon = torch.stack(final_recon, dim=1).mean(dim=1)

        # ===== 6. Contrastive Head =====
        out = self.decoder_to_contrastive(final_recon)

        return out

    # ======================
    # Encoding Mask
    # ======================
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        out_x[mask_nodes] += self.enc_mask_token

        src, dst = g.edges()
        use_g = dgl.graph(
            (src.clone(), dst.clone()),
            num_nodes=g.num_nodes(),
            device=g.device
        )

        return use_g, out_x, (mask_nodes, keep_nodes)

    # ======================
    # Remask
    # ======================
    def random_remask(self, g, rep, remask_rate=0.5):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)

        num_remask_nodes = int(remask_rate * num_nodes)

        remask_nodes = perm[:num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes