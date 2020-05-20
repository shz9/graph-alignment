"""
Author: Shadi Zabad
Date: April 2020

Note: Some of the operations here are copied from the examples/utilities code of torch_geometric.

"""

from multiprocessing import Pool
import networkx as nx
import numpy as np
import torch
import torch_geometric as tg
import math
import random
from parallel_betweenness import betweenness_centrality_parallel
import _pickle as cPickle
import bz2
import glob
import os
import errno


def add_negative_anchor_edge_index(tg_dataset):
    for i, g1 in enumerate(tg_dataset):
        for j, g2 in enumerate(tg_dataset):
            if i != j:
                g1.anchor_data[g2.gidx]['negative_anchor_edge_index'] = obtain_negative_anchors(
                    g1.x.size(0),
                    g2.x.size(0),
                    g1.anchor_data[g2.gidx]['anchor_edge_index']
                )


def pickle_data(data, output_file):
    with bz2.BZ2File(output_file, 'w') as f:
        cPickle.dump(data, f)


def unpickle_data(input_file):
    with bz2.BZ2File(input_file, 'rb') as inpf:
        return cPickle.load(inpf)


def save_tg_dataset(dataset, output_dir):

    for g1 in dataset:
        fname = os.path.join(output_dir, g1.graph_name + '.pbz2')
        print("Writing data file:", fname)
        pickle_data(g1, fname)


def load_tg_dataset(dataset_dir):

    tg_data = []

    for fname in glob.glob(os.path.join(dataset_dir, '*.pbz2')):
        tg_data.append(unpickle_data(fname))

    return tg_data


def make_dirs(new_dir):
    try:
        os.makedirs(new_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def select_top_k_anchors(tg_data, k=3, normalize_dist=True):

    for g1 in tg_data:
        for g2 in g1.anchor_data.keys():

            ei_nodes = []  # edge index nodes
            ei_anchors = []  # edge index anchors
            sp_dist = []  # short path distance

            for i, (eian, sp) in g1.anchor_data[g2]['closest_anchor_data'].items():

                ei_nodes += [i]*len(eian[:k])
                ei_anchors += eian[:k]
                dist = sp[:k]
                if normalize_dist:
                    dist = list(np.array(dist) / (sum(dist)))
                sp_dist += dist

            g1.anchor_data[g2]['closest_anchors'] = torch.stack([torch.Tensor(ei_nodes),
                                                                 torch.Tensor(ei_anchors)], dim=0).to(torch.long)
            g1.anchor_data[g2]['sp_dist'] = torch.Tensor(sp_dist).reshape(-1, 1)

    return tg_data


def get_anchor_link_labels(pos_anchor_idx, neg_anchor_idx, neg_label=0):

    link_labels = torch.ones(pos_anchor_idx.size(1) +
                              neg_anchor_idx.size(1)).float()

    link_labels *= neg_label

    link_labels[:pos_anchor_idx.size(1)] = 1.
    return link_labels


def get_central_nodes_in_neighborhood(g, n, q_hops=3):

    q_order_neighb = nx.single_source_shortest_path_length(g, n, cutoff=q_hops)

    sorted_nodes = sorted(q_order_neighb.items(),
                          key=lambda x: g.graph['centrality'][x[0]],
                          reverse=True)

    return [(list(g.nodes()).index(an), d) for an, d in sorted_nodes]


def get_closest_anchors_node(g, n, anchors, shuffle=True):

    anchor_dist = []

    if len(anchors) > 0:
        for an in anchors:

            try:
                dist = nx.shortest_path_length(g, source=n, target=an)
            except nx.exception.NetworkXNoPath:
                dist = np.inf

            anchor_dist.append((list(g.nodes()).index(an), dist))

        if shuffle:
            # Shuffle randomly to account for collisions
            np.random.shuffle(anchor_dist)

        anchor_dist = sorted(anchor_dist, key=lambda x: x[1])

        if all([np.isinf(d) for _, d in anchor_dist]):
            anchor_dist = get_central_nodes_in_neighborhood(g, n)

    else:
        anchor_dist = get_central_nodes_in_neighborhood(g, n)

    anchor_idx, dist = [list(t) for t in zip(*anchor_dist)]

    dist = list(1. / (np.array(dist) + 1))

    return {
        list(g.nodes()).index(n): (anchor_idx, dist)
    }


def get_closest_anchors(g, anchors, shuffle=True, n_proc=5):

    pool = Pool(n_proc)

    res = pool.starmap(get_closest_anchors_node, [(g, n, anchors, shuffle) for n in g.nodes()])

    pool.close()
    pool.join()

    return {k: v for d in res for k, v in d.items()}


def sample_negative_anchors(neg_anchor_edge_index, num_negative):

    perm = random.sample(range(neg_anchor_edge_index.size(1)),
                         min(num_negative, neg_anchor_edge_index.size(1)))
    perm = torch.tensor(perm).to(torch.long)

    return neg_anchor_edge_index[:, perm]


def obtain_negative_anchors(g1_n_nodes, g2_n_nodes, positive_anchors):

    row, col = positive_anchors

    neg_adj_mask = torch.ones(g1_n_nodes, g2_n_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    return neg_adj_mask.nonzero().t().to(torch.long)


def get_positive_anchors(g1, g2, test_ratio, shuffle=True):
    """

    :param g1: networkx graph 1
    :param g2: networkx graph 2
    :param test_ratio: test split ratio
    :param shuffle: flag to shuffle the anchor nodes

    """

    positive_anchors = list(set(g1.nodes()).intersection(set(g2.nodes())))

    g1_ei = [list(g1.nodes()).index(n) for n in positive_anchors]
    g2_ei = [list(g2.nodes()).index(n) for n in positive_anchors]

    anchor_ei = torch.stack([torch.Tensor(g1_ei), torch.Tensor(g2_ei)], dim=0).to(torch.long)

    if shuffle:
        perm = torch.tensor(random.sample(range(anchor_ei.size(1)), anchor_ei.size(1))).to(torch.long)

        positive_anchors = list(np.array(positive_anchors)[perm.numpy()])
        anchor_ei = anchor_ei[:, perm]

    if test_ratio is None or test_ratio == 0.0:

        return {
            'positive_anchors': positive_anchors,
            'train_positive_anchors': positive_anchors,
            'test_positive_anchors': positive_anchors,
            'anchor_edge_index': anchor_ei,
            'train_anchor_edge_index': anchor_ei,
            'test_anchor_edge_index': anchor_ei
        }

    else:

        n_t = int(math.floor(test_ratio*len(positive_anchors)))

        train_positive_anchors, test_positive_anchors = positive_anchors[n_t:], positive_anchors[:n_t]

        train_ei, test_ei = anchor_ei[:, n_t:], anchor_ei[:, :n_t]

        return {
            'positive_anchors': positive_anchors,
            'train_positive_anchors': train_positive_anchors,
            'test_positive_anchors': test_positive_anchors,
            'anchor_edge_index': anchor_ei,
            'train_anchor_edge_index': train_ei,
            'test_anchor_edge_index': test_ei
        }


def from_nx_to_tg_graphs(graphs, attributes=None, test_ratio=0.8, ntp_ratio=2, normalize_dist=True):

    print("> Transforming the data...")

    tg_graphs = []

    for i, g1 in enumerate(graphs):

        if g1.graph['centrality'] is None:
            g1.graph['centrality'] = betweenness_centrality_parallel(
                g1, processes=min(5, max(2, g1.number_of_nodes() // 1000))
            )
            pickle_data(g1.graph['centrality'], g1.graph['centrality_file'])


        tg_g1 = tg.utils.from_networkx(g1)

        try:
            tg_g1['graph_name'] = g1.graph['name']
        except KeyError:
            tg_g1['graph_name'] = 'g' + str(i)

        tg_g1['gidx'] = 'g' + str(i)
        tg_g1['anchor_data'] = {}

        # If the graph is not attributed, add the node degrees as attributes:
        if attributes is None:
            tg_g1.x = torch.Tensor(list(dict(g1.degree()).values())).reshape(-1, 1)
        else:
            tg_g1.x = torch.Tensor(attributes[i])

        for j, g2 in enumerate(graphs):

            print(i, j)

            if j == i:
                continue
            if j < i:

                tg_g1['anchor_data']['g' + str(j)] = {}

                for k, v in tg_graphs[j]['anchor_data']['g' + str(i)].items():
                    if 'positive_anchors' in k:
                        tg_g1['anchor_data']['g' + str(j)][k] = v
                    elif 'anchor_edge_index' in k:
                        tg_g1['anchor_data']['g' + str(j)][k] = v[[-1, 0], :]


            if j > i:

                tg_g1['anchor_data']['g' + str(j)] = get_positive_anchors(g1, g2, test_ratio)

                negative_anchor_edge_index = obtain_negative_anchors(
                    g1.number_of_nodes(),
                    g2.number_of_nodes(),
                    tg_g1['anchor_data']['g' + str(j)]['anchor_edge_index']
                )

                tg_g1['anchor_data']['g' + str(j)]['test_negative_anchor_edge_index'] = sample_negative_anchors(
                    negative_anchor_edge_index,
                    tg_g1['anchor_data']['g' + str(j)]['test_anchor_edge_index'].size(1)*ntp_ratio
                )

            tg_g1['anchor_data']['g' + str(j)]['closest_anchor_data'] = get_closest_anchors(
                g1,
                tg_g1['anchor_data']['g' + str(j)]['train_positive_anchors'],
                normalize_dist
            )

        tg_graphs.append(tg_g1)

    return tg_graphs
