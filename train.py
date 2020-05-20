"""
Author: Shadi Zabad
Date: April 2020
"""

import numpy as np
import torch
from utils import get_anchor_link_labels, sample_negative_anchors, obtain_negative_anchors
from early_stopping import EarlyStopping
from model import ContrastiveLoss, SiameseBCELoss


def validate(model, loss_func, g1, g2, criterion='loss'):

    model.eval()

    neg_label = 0.0
    if 'Cosine' in str(loss_func):
        neg_label = -1.0

    link_labels = get_anchor_link_labels(g1.anchor_data[g2.gidx]['test_anchor_edge_index'],
                                         g1.anchor_data[g2.gidx]['test_negative_anchor_edge_index'],
                                         neg_label)

    x1, x2 = model(g1, g2)

    if criterion == 'loss':
        vloss = loss_func(x1, x2, link_labels).item()
        return vloss
    else:
        raise NotImplementedError('The validation criterion %s is not implemented' % criterion)


def train(model, loss, graph_data, model_output_dir, epochs=100, lr=0.01, ntp_ratio=2, valid_criterion='loss'):

    print("> Training the model...")

    early_stopping = EarlyStopping(model_save_dir=model_output_dir,
                                   patience=max(10, int((epochs - (0.5*len(graph_data)**2))/3)),
                                   verbose=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    neg_label = 0.0

    if loss == 'contrastive':
        loss_func = ContrastiveLoss()
    elif loss == 'BCE':
        loss_func = SiameseBCELoss()
    elif loss == 'cosine':
        loss_func = torch.nn.CosineEmbeddingLoss(margin=3.0)
        neg_label = -1.0
    else:
        raise NotImplementedError("Loss function not implemented")

    for ep in range(epochs):

        training_loss = []
        valid_perf = []

        for i in range(len(graph_data)):
            for j in range(len(graph_data)):

                if i == j:
                    continue

                model.train()
                optimizer.zero_grad()

                g1, g2 = graph_data[i], graph_data[j]

                g1.anchor_data[g2.gidx]['train_negative_anchor_edge_index'] = sample_negative_anchors(g1.anchor_data[g2.gidx]['negative_anchor_edge_index'],
                                                                                                      g1.anchor_data[g2.gidx]['train_anchor_edge_index'].size(1)*ntp_ratio)

                x1, x2 = model(g1, g2)
                anchor_link_labels = get_anchor_link_labels(g1.anchor_data[g2.gidx]['train_anchor_edge_index'],
                                                            g1.anchor_data[g2.gidx]['train_negative_anchor_edge_index'],
                                                            neg_label)
                loss = loss_func(x1, x2, anchor_link_labels)

                training_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                valid_perf.append(validate(model, loss_func, g1, g2, valid_criterion))


        print("---> Epoch", ep,
              "| Training Loss:", np.nanmean(training_loss),
              "| Validation criterion:", np.nanmean(valid_perf))

        early_stopping(np.nanmean(valid_perf), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
