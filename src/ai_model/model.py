"""DrugResponseNet: per-omics encoders + GNN drug encoder fused by a Transformer.

Each omics modality and the drug become one token. A [CLS] token attends over
all of them; its final state is regressed to ln(IC50). The CLS attention
weights give a modality-level explanation for each prediction, and missing
modalities are excluded through the attention padding mask.
"""
import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, a_hat):
        return self.drop(torch.relu(self.norm(torch.bmm(a_hat, self.lin(x)))))


class DrugGNN(nn.Module):
    """Graph convolutional encoder over the atom graph of a drug molecule."""

    def __init__(self, atom_dim, hidden=128, out_dim=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList([GCNLayer(hidden, hidden, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(2 * hidden, out_dim)

    def forward(self, x, a_hat, mask):
        h = self.inp(x)
        for layer in self.layers:
            h = h + layer(h, a_hat)  # residual connection
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        return self.out(torch.cat([mean, mx], dim=-1))


class FusionBlock(nn.Module):
    """Pre-norm Transformer encoder block that also returns attention weights."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w


class DrugResponseNet(nn.Module):
    def __init__(self, modality_dims, atom_dim, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.modalities = list(modality_dims)
        self.encoders = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, d_model)
                )
                for m, d in modality_dims.items()
            }
        )
        self.drug_encoder = DrugGNN(atom_dim, out_dim=d_model, dropout=dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # Token type embedding: 0 = CLS, 1..M = modalities, M+1 = drug.
        self.type_emb = nn.Embedding(len(self.modalities) + 2, d_model)
        self.blocks = nn.ModuleList([FusionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, omics, omics_mask, drug_x, drug_adj, drug_mask, return_attention=False):
        """
        omics:      {modality: FloatTensor[B, d_m]}
        omics_mask: BoolTensor[B, M]  True where the modality is available
        drug_*:     padded molecular graph tensors [B, N, F], [B, N, N], [B, N]
        """
        batch = omics_mask.shape[0]
        tokens = [self.cls.expand(batch, -1, -1)]
        for m in self.modalities:
            tokens.append(self.encoders[m](omics[m]).unsqueeze(1))
        tokens.append(self.drug_encoder(drug_x, drug_adj, drug_mask).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        x = x + self.type_emb.weight.unsqueeze(0)

        always = torch.zeros(batch, 1, dtype=torch.bool, device=x.device)
        padding = torch.cat([always, ~omics_mask, always], dim=1)  # True = ignore token

        attn = None
        for block in self.blocks:
            x, attn = block(x, padding)
        pred = self.head(x[:, 0]).squeeze(-1)
        if return_attention:
            # Attention from CLS to [modalities..., drug] in the last block.
            return pred, attn[:, 0, 1:]
        return pred
