import os
import networkx as nx
import glob
import numpy as np
import pandas as pd
from utils import make_dirs
from itertools import product


def generate_subgraph(g, anchor_nodes, node_df):

    final_nodes = []

    for an in anchor_nodes:
        try:
            final_nodes += [n for n in g.neighbors(an)] + [an]
        except Exception:
            continue

    node_subset = node_df.loc[node_df['id'].isin(final_nodes)]
    node_subset_dict = dict(zip(node_subset['id'], node_subset['username']))

    new_sg = nx.Graph(g.subgraph(final_nodes))

    return nx.relabel_nodes(new_sg, node_subset_dict)


def encode_gender(s):
    if s == 'male':
        return pd.Series({'f1': 1., 'f2': 0.})
    elif s == 'female':
        return pd.Series({'f1': 0., 'f2': 1.})
    else:
        return pd.Series({'f1': 0., 'f2': 0.})


def create_attribute_files(attribute_file, nodes):

    att_df = pd.read_csv(attribute_file, index_col=0)[['id', 'gender']]
    att_df = pd.merge(pd.DataFrame({'nodes': nodes}),
                      att_df, how='left', left_on='nodes', right_on='id').drop_duplicates()

    att_df = pd.concat([att_df, att_df['gender'].apply(encode_gender)], axis=1)

    return att_df[['f1', 'f2']].values


def transform_cosnet_files_to_standard_format(g1_files, g2_files, mapping_file, output_dir):

    make_dirs(output_dir)

    map_df = pd.read_csv(mapping_file, sep="\s+", names=['g1', 'g2'])

    g1_nodes = pd.read_csv(g1_files['nodes'], sep='\t', names=['id', 'username'])
    g2_nodes = pd.read_csv(g2_files['nodes'], sep='\t', names=['id', 'username'])

    g1_nx = nx.read_edgelist(g1_files['edges'], nodetype=int)
    g2_nx = nx.read_edgelist(g2_files['edges'], nodetype=int)

    g1 = generate_subgraph(g1_nx, list(g1_nodes.loc[g1_nodes['username'].isin(map_df['g1']), 'id'].values), g1_nodes)
    g2 = generate_subgraph(g2_nx, list(g2_nodes.loc[g2_nodes['username'].isin(map_df['g2']), 'id'].values), g2_nodes)

    print(g1.number_of_nodes())
    print(g2.number_of_nodes())

    map_df = map_df.loc[map_df['g1'].isin(list(g1.nodes())) & map_df['g2'].isin(list(g2.nodes()))]

    if 'attributes' in g1_files:
        attr_mat = create_attribute_files(g1_files['attributes'], list(g1.nodes()))
        np.save(os.path.join(output_dir, g1_files['name'] + ".attr"), attr_mat)

    if 'attributes' in g2_files:
        attr_mat = create_attribute_files(g2_files['attributes'], list(g2.nodes()))
        np.save(os.path.join(output_dir, g2_files['name'] + ".attr"), attr_mat)


    map_dict = dict(zip(map_df['g1'], map_df['g2']))

    nx.relabel_nodes(g1, map_dict, copy=False)

    nx.write_edgelist(g1, os.path.join(output_dir, g1_files['name'] + ".edgelist"))
    nx.write_edgelist(g2, os.path.join(output_dir, g2_files['name'] + ".edgelist"))


transform_cosnet_files_to_standard_format({
        'name': 'flickr',
        'nodes': 'data/flickr_vs_lastfm/flickr/flickr.nodes',
        'edges': 'data/flickr_vs_lastfm/flickr/flickr.edges',
        'attributes': 'data/graph_attributes/multiple_flickr.csv'
    },
    {
        'name': 'lastfm',
        'nodes': 'data/flickr_vs_lastfm/lastfm/lastfm.nodes',
        'edges': 'data/flickr_vs_lastfm/lastfm/lastfm.edges',
        'attributes': 'data/graph_attributes/multiple_lastfm.csv'
    },
    "data/flickr_vs_lastfm/flickr-lastfm.map.raw",
    "data/flickr_vs_lastfm/edgelist_data")

