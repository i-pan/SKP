"""
Collection of heads which take as input a sequence of embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class TransformerHead(nn.Module):
    def __init__(self, cfg, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        seq_len = self.cfg.seq_len
        if self.cfg.seq2cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            seq_len += 1
        if cfg.add_positional_embedding:
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feature_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=self.cfg.transformer_nhead,
                dim_feedforward=self.cfg.transformer_dim_feedforward or feature_dim,
                # dropout applied in self-attention and FFN
                dropout=self.cfg.transformer_dropout or 0.0,
                activation=self.cfg.transformer_activation or "gelu",
                norm_first=self.cfg.transformer_norm_first or False,
                batch_first=True,
                bias=True,
            ),
            num_layers=self.cfg.transformer_num_layers or 1,
        )

        # dropout applied to elements of embedding
        self.dropout = nn.Dropout(self.cfg.dropout or 0.0)

        if self.cfg.seq2seq:
            # classifies each element in sequence
            self.classifier_seq = nn.Linear(
                feature_dim, self.cfg.seq_num_classes or self.cfg.num_classes
            )

        if self.cfg.seq2cls:
            # classifies sequence using cls_token
            self.classifier_cls = nn.Linear(
                feature_dim, self.cfg.cls_num_classes or self.cfg.num_classes
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            if mask is not None:
                # add mask for cls token
                mask = torch.cat(
                    [torch.ones((x.shape[0], 1), device=mask.device).bool(), mask],
                    dim=1,
                )
        if hasattr(self, "pos_embed"):
            x += self.pos_embed
        x = self.transformer(x, src_key_padding_mask=mask)
        out = {}
        if hasattr(self, "classifier_cls"):
            out["logits_cls"] = self.classifier_cls(self.dropout(x[:, 0]))
        if hasattr(self, "classifier_seq"):
            out["logits_seq"] = self.classifier_seq(self.dropout(x[:, 1:]))
        return out


class BasicAttention(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.0, version: str = "v1"):
        super().__init__()
        version = version.lower()
        if version == "v1":
            self.mlp = nn.Sequential(
                nn.Tanh(), nn.Dropout(dropout), nn.Linear(feature_dim, 1)
            )
        elif version == "v2":
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(x)
        a = a.softmax(dim=1)
        x = (x * a).sum(dim=1)
        return x


class GatedAttention(nn.Module):
    # From: https://arxiv.org/pdf/1802.04712.pdf
    def __init__(self, L: int, M: int, dropout: float = 0.0):
        super().__init__()
        assert L > M, f"L={L} must be greater than M={M}"
        self.U = nn.Linear(L, M, bias=False)
        self.V = nn.Linear(L, M, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Linear(M, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, num_features, feature_dim)
        b, n, f = x.size()
        x_reshape = rearrange(x, "b n d -> (b n) d")
        att1 = torch.sigmoid(self.U(x_reshape)) * torch.tanh(self.V(x_reshape))
        att1 = self.dropout(att1)
        att2 = self.w(att1)
        att2_reshape = rearrange(att2, "(b n) 1 -> b n 1", b=b, n=n)
        att = torch.nn.functional.softmax(att2_reshape, dim=1)
        out = torch.sum(att * x, 1)

        return out


def convert_seq_and_mask_to_packed_sequence(
    seq: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert seq.shape[0] == mask.shape[0]
    lengths = mask.sum(1)
    seq = nn.utils.rnn.pack_padded_sequence(
        seq, lengths.cpu().int(), batch_first=True, enforce_sorted=False
    )
    return seq


class BiRNNHead(nn.Module):
    def __init__(self, cfg, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        if self.cfg.apply_norm_first_to_features:
            self.norm = nn.LayerNorm(feature_dim)

        if self.cfg.rnn_type == "LSTM":
            rnn = nn.LSTM
        elif self.cfg.rnn_type == "GRU":
            rnn = nn.GRU
        else:
            raise Exception(
                f"`rnn_type` must be one of [`LSTM`, `GRU`], got `{self.cfg.rnn_type}`"
            )
        self.rnn = rnn(
            input_size=feature_dim,
            hidden_size=self.cfg.rnn_hidden_size or feature_dim // 2,
            num_layers=self.cfg.rnn_num_layers or 1,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.rnn_dropout or 0.0,
        )

        self.dropout = nn.Dropout(self.cfg.dropout or 0.0)

        rnn_out_dim = (
            self.cfg.rnn_hidden_size * 2
            if self.cfg.rnn_hidden_size is not None
            else feature_dim
        )
        if self.cfg.seq2cls:
            # use attention to aggregate features across sequence
            assert self.cfg.attention_type in ["basic", "gated"]
            modules = []
            if self.cfg.attention_type == "basic":
                modules.append(
                    BasicAttention(
                        rnn_out_dim,
                        dropout=self.cfg.attention_dropout or 0.0,
                        version=self.cfg.attention_version or "v1",
                    )
                )
            elif self.cfg.attention_type == "gated":
                modules.append(
                    GatedAttention(
                        rnn_out_dim,
                        rnn_out_dim // 4,
                        dropout=self.cfg.attention_dropout or 0.0,
                    )
                )
            modules.append(
                nn.Linear(
                    rnn_out_dim,
                    self.cfg.cls_num_classes or self.cfg.num_classes,
                )
            )
            self.classifier_cls = nn.Sequential(*modules)

        if self.cfg.seq2seq:
            self.classifier_seq = nn.Linear(
                rnn_out_dim,
                self.cfg.seq_num_classes or self.cfg.num_classes,
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.cfg.apply_norm_first_to_features:
            x = self.norm(x)

        input = x

        if mask is not None:
            # convert to PackedSequence
            L = x.shape[1]
            x = convert_seq_and_mask_to_packed_sequence(x, mask)

        x, _ = self.rnn(x)

        if mask is not None:
            # convert back to tensor
            x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=L)[0]

        if self.cfg.add_skip_connection:
            x = x + input

        out = {}
        if hasattr(self, "classifier_cls"):
            out["logits_cls"] = self.classifier_cls(self.dropout(x))
        if hasattr(self, "classifier_seq"):
            out["logits_seq"] = self.classifier_seq(self.dropout(x))
        return out


class DoubleSkipBiRNNHead(nn.Module):
    # Adapted from https://github.com/darraghdog/rsna/blob/master/scripts/trainlstm.py
    def __init__(self, cfg, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        if self.cfg.rnn_type == "LSTM":
            rnn = nn.LSTM
        elif self.cfg.rnn_type == "GRU":
            rnn = nn.GRU
        self.rnns = nn.ModuleList(
            [
                rnn(
                    input_size=feature_dim,
                    hidden_size=feature_dim // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=self.cfg.rnn_dropout or 0.0,
                )
            ]
            * 2
        )

        self.linears = nn.ModuleList([nn.Linear(feature_dim, feature_dim)] * 2)
        self.dropout = nn.Dropout(self.cfg.dropout or 0.0)

        if self.cfg.seq2cls:
            # use attention to aggregate features across sequence
            assert self.cfg.attention_type in ["basic", "gated"]
            modules = []
            rnn_out_dim = feature_dim
            if self.cfg.attention_type == "basic":
                modules.append(
                    BasicAttention(
                        rnn_out_dim,
                        dropout=self.cfg.attention_dropout or 0.0,
                        version=self.cfg.attention_version or "v1",
                    )
                )
            elif self.cfg.attention_type == "gated":
                modules.append(
                    GatedAttention(
                        rnn_out_dim,
                        rnn_out_dim // 4,
                        dropout=self.cfg.attention_dropout or 0.0,
                    )
                )
            modules.append(
                nn.Linear(
                    rnn_out_dim,
                    self.cfg.cls_num_classes or self.cfg.num_classes,
                )
            )
            self.classifier_cls = nn.Sequential(*modules)

        if self.cfg.seq2seq:
            self.classifier_seq = nn.Linear(
                feature_dim,
                self.cfg.seq_num_classes or self.cfg.num_classes,
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if mask is not None:
            L = x.shape[1]
            x = convert_seq_and_mask_to_packed_sequence(x, mask)

        h1, _ = self.rnns[0](x)
        h2, _ = self.rnns[1](h1)

        if mask is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=L)
            h1, _ = nn.utils.rnn.pad_packed_sequence(
                h1, batch_first=True, total_length=L
            )
            h2, _ = nn.utils.rnn.pad_packed_sequence(
                h2, batch_first=True, total_length=L
            )

        h_l1 = F.gelu(self.linears[0](h1))
        h_l2 = F.gelu(self.linears[1](h2))

        h = x + h1 + h2 + h_l1 + h_l2

        out = {}
        if hasattr(self, "classifier_cls"):
            out["logits_cls"] = self.classifier_cls(self.dropout(h))
        if hasattr(self, "classifier_seq"):
            out["logits_seq"] = self.classifier_seq(self.dropout(h))
        return out


if __name__ == "__main__":
    from skp.configs import Config
    from skp.toolbox.functions import count_parameters

    seq_len = 32
    dim_feat = 512

    cfg = Config()
    # cfg.add_cls_token = True
    # cfg.transformer_nhead = 16
    # cfg.transformer_num_layers = 2
    # cfg.transformer_dropout = 0.1
    # cfg.transformer_activation = "gelu"
    # cfg.transformer_dim_feedforward = dim_feat
    # cfg.transformer_norm_first = False
    # cfg.dropout = 0.1
    # cfg.seq_num_classes = 10
    # cfg.cls_num_classes = 20
    cfg.seq_len = seq_len
    cfg.seq2seq = True
    cfg.seq2cls = True

    # head = TransformerHead(cfg, dim_feat)

    cfg.rnn_dropout = 0.1
    cfg.seq_num_classes = 10
    cfg.cls_num_classes = 20
    cfg.rnn_type = "GRU"
    cfg.attention_type = "basic"
    cfg.attention_dropout = 0.1
    cfg.attention_version = "v2"
    cfg.add_skip_connection = True
    head = DoubleSkipBiRNNHead(cfg, dim_feat)
    x = torch.randn((2, seq_len, dim_feat))
    mask = torch.zeros((2, seq_len), dtype=torch.bool)
    out = head(x, mask)
    count_parameters(head)
    print(out["logits_seq"].shape, out["logits_cls"].shape)
