"""
Author: Shadi Zabad
Date: April 2020

"""

import pandas as pd
import numpy as np
import glob
import networkx as nx
import os
from utils import (make_dirs, from_nx_to_tg_graphs, save_tg_dataset,
                   load_tg_dataset, select_top_k_anchors, unpickle_data)


def transform_multiplex_networks(layer_info_file, edge_list_file, outdir, drop_weight=False):

    make_dirs(outdir)

    networks = pd.read_csv(layer_info_file, sep='\s+')
    edge_df = pd.read_csv(edge_list_file,
                          names=['layerID', 'source_node', 'target_node', 'weight'], sep='\s+')

    columns_to_output = ['source_node', 'target_node', 'weight']
    if drop_weight:
        columns_to_output = columns_to_output[:-1]

    graph_files = []

    for i, (net_id, net_name) in networks.iterrows():
        out_fname = os.path.join(outdir, net_name + ".edgelist")
        edge_df.loc[edge_df['layerID'] == net_id, columns_to_output].to_csv(
            out_fname,
            header=False,
            index=False,
            sep=' '
        )

        graph_files.append(out_fname)

    return graph_files


def read_arxiv_network_data(test_ratio, k_nearest,
                            input_dir="./data/arXiv/",
                            layers_file="arxiv_netscience_layers.txt",
                            edge_list_file="arxiv_netscience_multiplex.edges",
                            drop_weight=True):

    tg_data_dirname = os.path.join(input_dir, 'tg_datasets', 'tr_' + str(test_ratio))

    if os.path.isdir(tg_data_dirname):
        return select_top_k_anchors(load_tg_dataset(tg_data_dirname), k_nearest)


    graph_files = glob.glob(os.path.join(input_dir, 'edgelist_data', "*.edgelist"))

    if len(graph_files) < 1:
        graph_files = transform_multiplex_networks(os.path.join(input_dir, layers_file),
                                                   os.path.join(input_dir, edge_list_file),
                                                   os.path.join(input_dir, 'edgelist_data'),
                                                   drop_weight=drop_weight)

    graph_data = []

    for gf in graph_files:
        if drop_weight:
            g = nx.read_edgelist(gf)
        else:
            g = nx.read_edgelist(gf, data=(('weight',float),))

        g.graph['name'] = os.path.basename(gf).replace('.edgelist', '')
        g.graph['centrality_file'] = os.path.join(os.path.dirname(gf), g.graph['name'] + '.centrality')

        if os.path.isfile(g.graph['centrality_file']):
            g.graph['centrality'] = unpickle_data(g.graph['centrality_file'])
        else:
            g.graph['centrality'] = None

        graph_data.append(g)

    tg_data = from_nx_to_tg_graphs(graph_data, test_ratio=test_ratio)

    make_dirs(tg_data_dirname)
    save_tg_dataset(tg_data, tg_data_dirname)

    return select_top_k_anchors(tg_data, k_nearest)


def read_sacch_network_data(test_ratio, k_nearest,
                            input_dir="./data/SacchCere/",
                            layers_file="sacchcere_genetic_layers.txt",
                            edge_list_file="sacchcere_genetic_multiplex.edges",
                            drop_weight=True):

    tg_data_dirname = os.path.join(input_dir, 'tg_datasets', 'tr_' + str(test_ratio) + '.pkl')

    if os.path.isdir(tg_data_dirname):
        return select_top_k_anchors(load_tg_dataset(tg_data_dirname), k_nearest)

    graph_files = glob.glob(os.path.join(input_dir, 'edgelist_data', "*.edgelist"))

    if len(graph_files) < 1:
        graph_files = transform_multiplex_networks(os.path.join(input_dir, layers_file),
                                                   os.path.join(input_dir, edge_list_file),
                                                   os.path.join(input_dir, 'edgelist_data'),
                                                   drop_weight=drop_weight)

    graph_data = []

    for gf in graph_files:
        if drop_weight:
            g = nx.read_edgelist(gf, create_using=nx.DiGraph()).to_undirected()
        else:
            g = nx.read_edgelist(gf, create_using=nx.DiGraph(), data=(('weight',float),)).to_undirected()

        g.graph['name'] = os.path.basename(gf).replace('.edgelist', '')
        g.graph['centrality_file'] = os.path.join(os.path.dirname(gf), g.graph['name'] + '.centrality')

        if os.path.isfile(g.graph['centrality_file']):
            g.graph['centrality'] = unpickle_data(g.graph['centrality_file'])
        else:
            g.graph['centrality'] = None

        graph_data.append(g)

    tg_data = from_nx_to_tg_graphs(graph_data, test_ratio=test_ratio)

    make_dirs(tg_data_dirname)
    save_tg_dataset(tg_data, tg_data_dirname)

    return select_top_k_anchors(tg_data, k_nearest)


def read_synthetic_network_dataset(input_dir, test_ratio, k_nearest):

    tg_data_dirname = os.path.join(input_dir, 'tg_datasets', 'tr_' + str(test_ratio))

    if os.path.isdir(tg_data_dirname):
        return select_top_k_anchors(load_tg_dataset(tg_data_dirname), k_nearest)

    graph_files = glob.glob(os.path.join(input_dir, 'edgelist_data', "*.edgelist"))

    graph_data = []

    for gf in sorted(graph_files):
        g = nx.read_edgelist(gf)
        g.graph['name'] = os.path.basename(gf).replace('.edgelist', '')
        g.graph['centrality_file'] = os.path.join(os.path.dirname(gf), g.graph['name'] + '.centrality')

        if os.path.isfile(g.graph['centrality_file']):
            g.graph['centrality'] = unpickle_data(g.graph['centrality_file'])
        else:
            g.graph['centrality'] = None

        graph_data.append(g)

    tg_data = from_nx_to_tg_graphs(graph_data, test_ratio=test_ratio)

    make_dirs(tg_data_dirname)
    save_tg_dataset(tg_data, tg_data_dirname)

    return select_top_k_anchors(tg_data, k_nearest)


def read_flickr_lastfm_data(test_ratio, k_nearest, input_dir="data/flickr_vs_lastfm/"):

    tg_data_dirname = os.path.join(input_dir, 'tg_datasets', 'tr_' + str(test_ratio))

    if os.path.isdir(tg_data_dirname):
        return select_top_k_anchors(load_tg_dataset(tg_data_dirname), k_nearest)

    graph_files = glob.glob(os.path.join(input_dir, 'edgelist_data', "*.edgelist"))

    graph_data = []
    attributes = []

    for gf in graph_files:

        g = nx.read_edgelist(gf)
        g.graph['name'] = os.path.basename(gf).replace('.edgelist', '')
        g.graph['centrality_file'] = os.path.join(os.path.dirname(gf), g.graph['name'] + '.centrality')

        if os.path.isfile(g.graph['centrality_file']):
            g.graph['centrality'] = unpickle_data(g.graph['centrality_file'])
        else:
            g.graph['centrality'] = None

        attributes.append(np.load(gf.replace(".edgelist", ".attr.npy")))

        graph_data.append(g)

    tg_data = from_nx_to_tg_graphs(graph_data, attributes=attributes, test_ratio=test_ratio)

    make_dirs(tg_data_dirname)
    save_tg_dataset(tg_data, tg_data_dirname)

    return select_top_k_anchors(tg_data, k_nearest)

