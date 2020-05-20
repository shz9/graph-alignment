"""
Author: Shadi Zabad
Date: April 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class SiameseGNN(torch.nn.Module):

    def __init__(self, input_dim, layer_type='GCN', hidden_dim=2, output_dim=2, normalize=True):
        super(SiameseGNN, self).__init__()

        self.normalize_embed = normalize

        if layer_type == 'GCN':
            gnn_layer = GCNConv
        elif layer_type == 'GAT':
            gnn_layer = GATConv
        else:
            raise NotImplementedError('Layer type not supported')

        self.conv1 = gnn_layer(input_dim, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)
        self.conv2 = gnn_layer(hidden_dim, output_dim)

    def embed_graph(self, g):

        x, edge_index = g.x, g.edge_index

        x = self.prelu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        if self.normalize_embed:
            return F.normalize(x, p=2, dim=-1)
        else:
            return x

    def forward(self, g1, g2, edge_index_labels=None):

        x1 = self.embed_graph(g1)
        x2 = self.embed_graph(g2)

        if edge_index_labels is None:
            if self.training:
                pos_anchor_idx = g1.anchor_data[g2.gidx]['train_anchor_edge_index']
                neg_anchor_idx = g1.anchor_data[g2.gidx]['train_negative_anchor_edge_index']
            else:
                pos_anchor_idx = g1.anchor_data[g2.gidx]['test_anchor_edge_index']
                neg_anchor_idx = g1.anchor_data[g2.gidx]['test_negative_anchor_edge_index']
        else:
            pos_anchor_idx = g1.anchor_data[g2.gidx][edge_index_labels[0]]
            neg_anchor_idx = g1.anchor_data[g2.gidx][edge_index_labels[1]]


        total_anchor_edge_index = torch.cat([pos_anchor_idx, neg_anchor_idx], dim=-1)

        x1 = torch.index_select(x1, 0, total_anchor_edge_index[0])
        x2 = torch.index_select(x2, 0, total_anchor_edge_index[1])

        return x1, x2


class AnchoredSiameseGNN(torch.nn.Module):

    def __init__(self, input_dim, layer_type='GCN', hidden_dim=2, output_dim=2, normalize=True):
        super(AnchoredSiameseGNN, self).__init__()

        self.normalize_embed = normalize

        if layer_type == 'GCN':
            gnn_layer = GCNConv
        elif layer_type == 'GAT':
            gnn_layer = GATConv
        else:
            raise NotImplementedError('Layer type not supported')

        self.conv1 = gnn_layer(input_dim, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)
        self.conv2 = gnn_layer(hidden_dim, output_dim)

    def embed_graph(self, g):

        x, edge_index = g.x, g.edge_index

        x = self.prelu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        if self.normalize_embed:
            return F.normalize(x, p=2, dim=-1)
        else:
            return x

    def anchor_node_embeddings(self, x, anchors_edge_index, sp_dist):

        x_nodes = torch.index_select(x, 0, anchors_edge_index[0])
        x_anchors = torch.index_select(x, 0, anchors_edge_index[1])

        anc_x = torch.cat([x_nodes, x_anchors], dim=-1) * sp_dist

        labels = anchors_edge_index[0].view(anchors_edge_index[0].size(0), 1).expand(-1, anc_x.size(1))

        unique_labels = labels.unique(dim=0)

        x = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, anc_x)

        return x


    def forward(self, g1, g2, edge_index_labels=None):

        x1 = self.anchor_node_embeddings(self.embed_graph(g1),
                                         g1.anchor_data[g2.gidx]['closest_anchors'],
                                         g1.anchor_data[g2.gidx]['sp_dist'])

        x2 = self.anchor_node_embeddings(self.embed_graph(g2),
                                         g2.anchor_data[g1.gidx]['closest_anchors'],
                                         g2.anchor_data[g1.gidx]['sp_dist'])

        if edge_index_labels is None:
            if self.training:
                pos_anchor_idx = g1.anchor_data[g2.gidx]['train_anchor_edge_index']
                neg_anchor_idx = g1.anchor_data[g2.gidx]['train_negative_anchor_edge_index']
            else:
                pos_anchor_idx = g1.anchor_data[g2.gidx]['test_anchor_edge_index']
                neg_anchor_idx = g1.anchor_data[g2.gidx]['test_negative_anchor_edge_index']
        else:
            pos_anchor_idx = g1.anchor_data[g2.gidx][edge_index_labels[0]]
            neg_anchor_idx = g1.anchor_data[g2.gidx][edge_index_labels[1]]


        total_anchor_edge_index = torch.cat([pos_anchor_idx, neg_anchor_idx], dim=-1)

        x1 = torch.index_select(x1, 0, total_anchor_edge_index[0])
        x2 = torch.index_select(x2, 0, total_anchor_edge_index[1])

        return x1, x2


class ContrastiveLoss(nn.Module):
    """

    Modified from: https://github.com/adambielski/siamese-triplet
    to match Equation (7) in https://grlearning.github.io/papers/33.pdf

    - - -

    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=1., pos_margin=0.1, neg_margin=5.):
        super(ContrastiveLoss, self).__init__()

        if pos_margin is None:
            self.pos_margin = margin
        else:
            self.pos_margin = pos_margin

        if neg_margin is None:
            self.neg_margin = margin
        else:
            self.neg_margin = neg_margin

    def forward(self, x1, x2, label, size_average=True):

        euc_dist = F.pairwise_distance(x1, x2, keepdim=True)

        cont_loss = (label*torch.clamp(euc_dist - self.pos_margin, min=0.0) +
                     (1. - label)*torch.clamp(self.neg_margin - euc_dist, min=0.0))

        return torch.mean(cont_loss)


class SiameseBCELoss(nn.Module):

    def __init__(self):
        super(SiameseBCELoss, self).__init__()

    def forward(self, x1, x2, label, size_average=True):

        prod = torch.einsum("ef,ef->e", x1, x2)
        return F.binary_cross_entropy_with_logits(prod, label)
