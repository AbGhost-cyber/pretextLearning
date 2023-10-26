import torch.distributed as dist
from byol_pytorch import BYOL

from torch import nn
import torch
import torch.nn.functional as F


def default(val, def_val):
    return def_val if val is None else val


def MaybeSyncBatchNorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


BYOL()