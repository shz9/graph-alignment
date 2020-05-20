"""
Author: Shadi Zabad
Date: April 2020
"""

import numpy as np
import pandas as pd
import os
import torch
from multiprocessing import Pool


def soft_ordering(total_edge_index, sim_scores):

    df = pd.DataFrame(np.concatenate((total_edge_index, sim_scores.reshape(1, -1))).T,
                      columns=['source', 'target', 'score'])

    df['source'] = df['source'].astype(np.int)
    df['target'] = df['target'].astype(np.int)

    df = df.sort_values(by=['source', 'score'], ascending=False)

    target_order = {}

    for s in df['source'].unique():
        sorted_targets = df.loc[df['source'] == s, 'target']
        target_order[s] = list(sorted_targets)

    return target_order


def greedy_matching(total_edge_index, sim_scores):

    pred_pairs = []

    while total_edge_index.shape[1] > 0:
        pair_idx = np.argmax(sim_scores)

        a, b = total_edge_index[:, pair_idx].flatten()
        pred_pairs.append((a, b))

        step_filt = (total_edge_index[0, :] != a) & (total_edge_index[1, :] != b)

        total_edge_index = total_edge_index[:, step_filt]
        sim_scores = sim_scores[step_filt]

    return pred_pairs


def accuracy(pos_anchors, pred_anchors):
    """
    Defined as in Equation (13) of Trung et al.
    https://www.sciencedirect.com/science/article/pii/S0957417419305937

    :param pos_edge_index: list of tuples of matching nodes in graphs 1 and 2
    :param pred_edge_index: list of tuples of predicted matching nodes in graphs 1 and 2

    """
    return float(len(set(pos_anchors).intersection(set(pred_anchors)))) / len(pos_anchors)


def precision(pos_anchors, soft_ordered_pred, k=10):

    num_succ = sum([1 for s, t in pos_anchors if soft_ordered_pred[s].index(t) < k])

    return float(num_succ) / len(pos_anchors)


def MAP(pos_anchors, soft_ordered_pred):

    return np.mean([1. / (soft_ordered_pred[s].index(t) + 1) for s, t in pos_anchors])


def evaluate_model_pairwise(model, loss, g1, g2, prec_k=(1, 3, 5, 10, 30), compute_accuracy=False):

    model.eval()

    metrics = {}

    metrics['pair'] = g1.graph_name + '-->' + g2.graph_name

    pos_anchor_idx = g1.anchor_data[g2.gidx]['anchor_edge_index']
    neg_anchor_idx = g1.anchor_data[g2.gidx]['negative_anchor_edge_index']

    print(">>> Evaluating model...")
    x1, x2 = model(g1, g2, edge_index_labels=('anchor_edge_index', 'negative_anchor_edge_index'))

    if loss == 'contrastive':
        euc_dist = torch.pairwise_distance(x1, x2, keepdim=True)
        model_pred = 1. / (1. + euc_dist)
    elif loss == 'cosine':
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_pred = 1. - 0.5 * (1. - cos(x1, x2))
    elif loss == 'BCE':
        model_pred = torch.sigmoid(torch.einsum("ef,ef->e", x1, x2))
    else:
        raise NotImplementedError("Loss function not implemented")

    model_pred = model_pred.detach().numpy()
    anchor_idx = torch.cat([pos_anchor_idx, neg_anchor_idx], dim=-1).detach().numpy()

    true_anchor_idx = list(map(tuple, pos_anchor_idx.numpy().T))

    ####### Computing Accuracy #######
    if compute_accuracy:
        print(">>> Computing accuracy...")
        pred_anchor_idx = greedy_matching(anchor_idx, model_pred)
        metrics['accuracy'] = accuracy(true_anchor_idx, pred_anchor_idx)

    ####### Computing Precision #######
    print(">>> Computing precision metrics...")
    ordered_sim = soft_ordering(anchor_idx, model_pred)

    for k in prec_k:
        metrics['precision@' + str(k)] = precision(true_anchor_idx, ordered_sim, k)

    metrics['MAP'] = MAP(true_anchor_idx, ordered_sim)

    return metrics


def evaluate_model(model, loss, graph_data, output_dir, compute_accuracy=True):

    print("> Evaluating model performance...")

    res_metrics = []

    for i, g1 in enumerate(graph_data):
        for j, g2 in enumerate(graph_data):
            if i != j:
                res_metrics.append(evaluate_model_pairwise(model, loss, g1, g2, compute_accuracy=compute_accuracy))

    met_df = pd.DataFrame(res_metrics)
    met_df.to_csv(os.path.join(output_dir, 'paired_metrics.csv'))
    met_df.describe().to_csv(os.path.join(output_dir, 'summary_metrics.csv'))
