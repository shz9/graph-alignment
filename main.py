"""
Author: Shadi Zabad
Date: April 2020
"""

from utils import make_dirs, add_negative_anchor_edge_index
from read_transform_data import (read_sacch_network_data,
                                 read_arxiv_network_data,
                                 read_synthetic_network_dataset,
                                 read_flickr_lastfm_data)
from evaluation import evaluate_model
from model import SiameseGNN, AnchoredSiameseGNN
from train import train
import torch
import os
import glob


def train_and_evaluate_model(dataset='arXiv',
                             model_name='SiameseGNN',
                             model_save_dir='./saved_model',
                             metrics_save_dir='./metrics',
                             k_nearest=3,
                             epochs=100,
                             test_ratio=0.8,
                             loss='contrastive',
                             valid_criterion='loss',
                             compute_accuracy=True):

    model_output_dir = os.path.join(model_save_dir, dataset,
                                    model_name + '_' + loss + '_tr_' + str(test_ratio) + '_kn_' + str(k_nearest))
    make_dirs(model_output_dir)

    metrics_output_dir = os.path.join(metrics_save_dir, dataset,
                                      model_name + '_' + loss + '_tr_' + str(test_ratio) + '_kn_' + str(k_nearest))
    make_dirs(metrics_output_dir)

    print("> Reading the data at %s..." % dataset)

    if dataset == 'SacchCere':
        g_data = read_sacch_network_data(test_ratio, k_nearest)
    elif dataset == 'arXiv':
        g_data = read_arxiv_network_data(test_ratio, k_nearest)
    elif dataset == 'flickr_vs_lastfm':
        g_data = read_flickr_lastfm_data(test_ratio, k_nearest)
    elif 'synthetic' in dataset:
        g_data = read_synthetic_network_dataset(dataset, test_ratio, k_nearest)
    else:
        raise NotImplementedError('Training and inference on dataset %s is not implemented' % dataset)

    if model_name == 'SiameseGNN':
        model = SiameseGNN(g_data[0].x.shape[1], normalize=True)
    elif model_name == 'AnchoredSiameseGNN':
        model = AnchoredSiameseGNN(g_data[0].x.shape[1], normalize=True)
    else:
        raise NotImplementedError('Model "%s" is not implemented' % model_name)

    add_negative_anchor_edge_index(g_data)

    if os.path.isfile(os.path.join(model_output_dir, 'checkpoint.pt')):
        model.load_state_dict(torch.load(os.path.join(model_output_dir, 'checkpoint.pt')))
    else:
        train(model, loss, g_data, model_output_dir, epochs=epochs, valid_criterion=valid_criterion)


    evaluate_model(model, loss, g_data, metrics_output_dir, compute_accuracy=compute_accuracy)

    print("> Done!")
    print("Model can be found at: %s" % model_output_dir)
    print("Metrics can be found at: %s" % metrics_output_dir)

if __name__ == '__main__':

    # Apply on ArXiv dataset:

    train_and_evaluate_model("arXiv", model_name='SiameseGNN',
                             test_ratio=0.9, loss='cosine', compute_accuracy=False)
    train_and_evaluate_model("arXiv", model_name='AnchoredSiameseGNN',
                             test_ratio=0.9, loss='cosine', compute_accuracy=False)
    """

    # test ratio:

    for tr in [0.5, 0.7, 0.9]:
        train_and_evaluate_model("data/synthetic/PA/n_100/erp_0.1/nrp_0.1/",
                                 model_name='SiameseGNN', test_ratio=tr, loss='cosine')
        train_and_evaluate_model("data/synthetic/PA/n_100/erp_0.1/nrp_0.1/",
                                 model_name='AnchoredSiameseGNN', test_ratio=tr, loss='cosine')

    # edge removal probabilities (synthetic):
    for erpf in glob.glob("data/synthetic/PA/n_100/*/nrp_0.1/"):
        train_and_evaluate_model(erpf,
                                 model_name='SiameseGNN', test_ratio=0.8, loss='cosine')
        train_and_evaluate_model(erpf,
                                 model_name='AnchoredSiameseGNN', test_ratio=0.8, loss='cosine')

    # node removal probabilities (synthetic):
    for nrpf in glob.glob("data/synthetic/PA/n_100/erp_0.1/*/"):
        train_and_evaluate_model(nrpf,
                                 model_name='SiameseGNN', test_ratio=0.8, loss='cosine')
        train_and_evaluate_model(nrpf,
                                 model_name='AnchoredSiameseGNN', test_ratio=0.8, loss='cosine')

    # graph type:
    for gtf in glob.glob("data/synthetic/*/n_100/erp_0.1/nrp_0.1/"):
        train_and_evaluate_model(gtf,
                                 model_name='SiameseGNN', test_ratio=0.8, loss='cosine')
        train_and_evaluate_model(gtf,
                                 model_name='AnchoredSiameseGNN', test_ratio=0.8, loss='cosine')

    # graph size:
    for gsf in glob.glob("data/synthetic/PA/*/erp_0.1/nrp_0.1/"):
        train_and_evaluate_model(gsf,
                                 model_name='SiameseGNN', test_ratio=0.8, loss='cosine')
        train_and_evaluate_model(gsf,
                                 model_name='AnchoredSiameseGNN', test_ratio=0.8, loss='cosine')

    # Apply on flickr vs lastfm dataset:
    train_and_evaluate_model("flickr_vs_lastfm", model_name='SiameseGNN',
                             test_ratio=0.9, loss='cosine', compute_accuracy=False)
    train_and_evaluate_model("flickr_vs_lastfm", model_name='AnchoredSiameseGNN',
                             test_ratio=0.9, loss='cosine', compute_accuracy=False)
    """
