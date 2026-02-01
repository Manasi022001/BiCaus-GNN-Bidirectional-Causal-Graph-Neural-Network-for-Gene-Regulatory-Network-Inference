"""
Baseline GNN model for comparison
Simple heterogeneous GNN using GraphSAGE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class BaselineGNN(nn.Module):
    """
    Simple heterogeneous GNN for CpG-gene correlation prediction
    Uses GraphSAGE convolutions
    """
    def __init__(self, cpg_in_channels, gene_in_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        # [COPY FROM NOTEBOOK: lines ~2343-2422]
        pass
