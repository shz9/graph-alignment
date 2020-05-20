"""
Author: Shadi Zabad
Date: April 2020
"""

import networkx as nx
import numpy as np
import random
import os
from utils import make_dirs


def complete_disconnected_graph(g):
    g_comp = list(nx.connected_components(g))

    while len(g_comp) > 1:
        g.add_edge(random.sample(g_comp[0], 1)[0], random.sample(g_comp[1], 1)[0])
        g_comp = list(nx.connected_components(g))


def generate_synthetic_graphs(n_graphs=2,
                              n_nodes=100,
                              main_graph="ER",
                              edge_removal_prob=0.1,
                              node_removal_prob=0.1):
    graphs = []

    if main_graph == "WS":
        g0 = nx.watts_strogatz_graph(n=n_nodes, k=5, p=0.85)
    elif main_graph == "ER":
        g0 = nx.erdos_renyi_graph(n=n_nodes, p=0.15)
    elif main_graph == "PA":
        g0 = nx.barabasi_albert_graph(n=n_nodes, m=n_nodes // 20)
    else:
        raise NotImplementedError

    # Complete graph (in case of isolates or disconnected components):
    complete_disconnected_graph(g0)

    graphs.append(g0)

    for i in range(n_graphs - 1):
        edge_prob = np.random.binomial(1, edge_removal_prob, size=g0.number_of_edges()).astype(bool)
        sampled_edges = np.array(list(g0.edges()))[~edge_prob]

        ng = nx.Graph()
        ng.add_edges_from(sampled_edges)

        node_prob = np.random.binomial(1, node_removal_prob, size=ng.number_of_nodes()).astype(bool)
        sampled_nodes = np.array(list(ng.nodes()))[node_prob]

        ng.remove_nodes_from(sampled_nodes)

        # Complete graph (in case of isolates or disconnected components):
        complete_disconnected_graph(ng)

        graphs.append(ng)

    return graphs


if __name__ == '__main__':

    main_dir = "./data/synthetic/"

    graph_types = ['ER', 'PA', 'WS']
    n_nodes = [50, 100, 1000]
    edge_removal_prob = [0.05, 0.1, 0.2, 0.3]
    node_removal_prob = [0.05, 0.1, 0.2, 0.3]

    for gt in graph_types:
        for n in n_nodes:
            for erp in edge_removal_prob:
                for nrp in node_removal_prob:

                    output_dir = os.path.join(main_dir, gt, 'n_' + str(n),
                                              'erp_' + str(erp), 'nrp_' + str(nrp), 'edgelist_data')
                    make_dirs(output_dir)

                    graphs = generate_synthetic_graphs(n_graphs=6, n_nodes=n, main_graph=gt,
                                                       edge_removal_prob=erp, node_removal_prob=nrp)

                    for i, g in enumerate(graphs):
                        nx.write_edgelist(g, os.path.join(output_dir, 'g' + str(i) + '.edgelist'))
