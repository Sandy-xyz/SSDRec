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
        """
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.attn_fc = nn.Linear(in_dim * 2, 1, bias=False)
        self.fc = nn.Linear(in_dim, in_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [E, 2d]
        e = self.leaky_relu(self.attn_fc(z2))  # [E, 1]
        return {'e': e}

    def forward(self, g, x):
        with g.local_scope():
            z = self.fc(x)  # [N, d]
            g.ndata['z'] = z

            g.apply_edges(self.edge_attention)

            a = dgl.nn.functional.edge_softmax(g, g.edata['e'])
            g.edata['a'] = a
            g.update_all(
                dgl.function.u_mul_e('z', 'a', 'm'),  # m_ij = α_ij * z_i
                dgl.function.sum('m', 'h_neigh')  # h_j = Σ α_ij z_i
            )

            Ax_tilde = g.ndata['h_neigh']  # [N, d]
            alpha = torch.sigmoid(self.alpha)
            x_smooth = x - alpha * (x - Ax_tilde)

        return x_smooth


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

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.encoder = nn.ModuleList()
        for _ in range(enc_num_layer):
            self.encoder.append(
                GraphConv(
                    hidden_dim,
                    hidden_dim,
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True
                )
            )
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
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.decoder_to_contrastive = nn.Linear(hidden_dim, in_dim, bias=False)

        self.reset_parameters()

        self.spectral_denoiser = AdaptiveSpectralDenoiser(in_dim=hidden_dim)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.input_proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.decoder_to_contrastive.weight, gain=1.414)

    def forward(self, g, x, drop_g1=None, is_item=1):

        pre_use_g, mask_x, _ = self.encoding_mask_noise(
            g, x, self.mask_rate
        )
        use_g = drop_g1 if drop_g1 is not None else g
        h = self.input_proj(mask_x)
        all_embeddings = []
        for layer in self.encoder:
            h = layer(use_g, h)
            all_embeddings.append(h)

        enc_rep = torch.stack(all_embeddings, dim=1).mean(dim=1)
        if is_item:
            enc_rep = self.spectral_denoiser(use_g, enc_rep)
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
        out = self.decoder_to_contrastive(final_recon)

        return out

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