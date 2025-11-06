import math
from typing import Optional

import torch
from torch import nn
    

class SeqEmbeddingLearned(nn.Module):
    """Learnable sequence position embeddings."""

    def __init__(self, num_pos_feats: int, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(num_pos_feats, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings with normal distribution."""
        nn.init.normal_(self.embed.weight)

    def forward(self, ln: int) -> torch.Tensor:
        """Return positional embeddings for sequence of length ln."""
        return self.embed.weight[:ln].unsqueeze(1)


class SeqEmbeddingSine(nn.Module):
    """Sinusoidal sequence position embeddings."""

    def __init__(self, max_len: int = 200, d_model: int = 512):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        te = torch.zeros(max_len, 1, d_model)
        te[:, 0, 0::2] = torch.sin(position * div_term)
        te[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("te", te)

    def forward(self, ln: int) -> torch.Tensor:
        """Return positional embeddings for sequence of length ln."""
        pos_t = self.te[:ln]
        return pos_t
    

class PositionEmbeddingLearned(nn.Module):
    """Learnable 2D position embeddings for spatial features."""

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings with uniform distribution."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        """Generate 2D position embeddings for spatial feature map."""
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # Embed x and y coordinates
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # Combine x and y embeddings
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos