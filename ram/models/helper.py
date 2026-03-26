import torch
import logging
import torch
import torch.nn as nn
from transformers import modeling_utils

def find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    """
    Finds the heads and the flattened indices to keep, taking already-pruned heads
    into account.

    Parameters
    ----------
    heads : list[int]
        Head indices requested for pruning.
    n_heads : int
        Total number of attention heads before this pruning step.
    head_size : int
        Size of each attention head.
    already_pruned_heads : set[int]
        Heads that were pruned in earlier steps.

    Returns
    -------
    tuple[set[int], torch.LongTensor]
        (new_heads_to_prune, flattened_indices_to_keep)
    """
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)

    heads = set(heads) - already_pruned_heads

    for head in heads:
        # Shift the head index left by however many smaller heads
        # were already removed earlier.
        shifted_head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[shifted_head] = False

    index = torch.arange(n_heads * head_size)[mask.view(-1)].long()
    return heads, index

# ------------------------------------------------------------------------------
# 1. Define Fallback Implementations
#    (Mirrors logic removed in transformers v4.45+)
# ------------------------------------------------------------------------------

def _prune_linear_layer(layer, index, dim=0):
    """Fallback implementation for prune_linear_layer"""
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 0:
            b = layer.bias.index_select(dim, index).clone().detach()
        else:
            b = layer.bias.clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def _prune_layer(layer, index, dim=None):
    """Fallback for prune_layer (generic)"""
    if isinstance(layer, nn.Linear):
        return _prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    
    # Handle Conv1D if available
    try:
        from transformers.pytorch_utils import prune_conv1d_layer
        if hasattr(modeling_utils, "Conv1D") and isinstance(layer, modeling_utils.Conv1D):
            return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    except ImportError:
        pass
        
    return _prune_linear_layer(layer, index, dim=0 if dim is None else dim)

def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Fallback for finding heads to prune"""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

# ------------------------------------------------------------------------------
# 2. Inject Missing Functions into transformers.modeling_utils
# ------------------------------------------------------------------------------

# Patch 'Conv1D' if missing
if not hasattr(modeling_utils, "Conv1D"):
    try:
        from transformers.pytorch_utils import Conv1D
        modeling_utils.Conv1D = Conv1D
    except ImportError:
        pass

# Patch 'prune_linear_layer'
if not hasattr(modeling_utils, "prune_linear_layer"):
    try:
        from transformers.pytorch_utils import prune_linear_layer
        modeling_utils.prune_linear_layer = prune_linear_layer
    except ImportError:
        modeling_utils.prune_linear_layer = _prune_linear_layer

# Patch 'prune_layer' (Critical for MeshGraphormer)
if not hasattr(modeling_utils, "prune_layer"):
    modeling_utils.prune_layer = _prune_layer

# Patch 'find_pruneable_heads_and_indices'
if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices
        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    except ImportError:
        modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

logging.info("\033[32m[Transformers Fix] Successfully injected missing pruning functions for MeshGraphormer compatibility.\033[0m")

NODE_CLASS_MAPPINGS = {}