"""Backbones used by the minimal reproduction."""

from .staeformer import STAEformer
from .stgcn import STGCN, STGCNSpatialServer

__all__ = ["STAEformer", "STGCN", "STGCNSpatialServer"]
