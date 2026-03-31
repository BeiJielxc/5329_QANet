import torch.nn as nn

from .layernorm import LayerNorm
from .groupnorm import GroupNorm


# Registry: norm_name -> class
normalizations = {
    "layer_norm": LayerNorm,
    "group_norm":  GroupNorm,
}


class _ChannelLayerNorm(nn.Module):
    """Wraps LayerNorm to normalize over the channel dim of [B, C, L] tensors.

    Standard LayerNorm operates on the last dimension; this wrapper transposes
    so that the channel dimension C becomes last, applies LN, then transposes
    back.  This matches the standard Transformer pre-norm convention where each
    position is normalized independently over its feature vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def get_norm(name: str, d_model: int, length: int, num_groups: int = 8) -> nn.Module:
    """
    Instantiate a normalization module by registry name.

    Args:
        name:       one of "layer_norm", "group_norm"
        d_model:    number of channels (C)
        length:     sequence length (L); kept for API compat, unused by layer_norm
        num_groups: number of groups; used only by group_norm

    Returns:
        nn.Module instance of the requested normalization.

    Shapes:
        "layer_norm" → _ChannelLayerNorm(d_model)
            normalizes over the channel dim C of [B, C, L], each position independent
        "group_norm"  → GroupNorm(num_groups, d_model)
            normalizes over [C/G, *spatial] per group of [B, d_model, *]
    """
    if name not in normalizations:
        raise ValueError(
            f"Unknown normalization '{name}'. Available: {list(normalizations.keys())}"
        )
    if name == "layer_norm":
        return _ChannelLayerNorm(d_model)
    else:  # group_norm
        return GroupNorm(num_groups, d_model)
