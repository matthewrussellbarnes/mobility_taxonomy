import networkx as nx
import os
import csv
import sys
import numpy as np
import scipy.stats as stats

import utilities


def create_artificial_network(probs_list=['random'], initial_n=10, iterations=2000, gamma_a=1.5, gamma_scale=2, degree_base_p=1, edge_increase_per_iteration=3, node_increase_on_iteration=1, node_increase_per_iteration=1, n_node_gammas_to_change=1, f_name_modifier=None):
    print(probs_list)

    path = os.path.join(utilities.dataset_path,
                        f"Artificial/%{f_name_modifier}artificial_{'_'.join(probs_list)}_i{iterations}.csv")
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f, ['n1', 'n2', 'creation_time'], delimiter=' ')
        writer.writeheader()

        G = nx.Graph()

        probs_dist_dict = {}
        G.add_nodes_from(list(range(initial_n)))
        for i in range(iterations):
            n = G.number_of_nodes()
            for p in probs_list:
                if p == 'random':
                    probs_dist_dict[p] = equal_probs(n)
                elif p == 'preferential_attachment':
                    probs_dist_dict[p] = degree_probs(
                        G.degree, n, degree_base_p)
                elif p == 'fitness':
                    gamma_dist = probs_dist_dict[p] if p in probs_dist_dict else gamma_probs(
                        gamma_a, gamma_scale, n)
                    probs_dist_dict[p] = static_gamma_probs(
                        gamma_a, gamma_scale, n, gamma_dist)
                elif p == 'gamma_every_i':
                    probs_dist_dict[p] = gamma_probs(
                        gamma_a, gamma_scale, n)
                elif p == 'gamma_n_change':
                    gamma_dist = probs_dist_dict[p] if p in probs_dist_dict else gamma_probs(
                        gamma_a, gamma_scale, n)
                    probs_dist_dict[p] = static_gamma_with_individual_changes(
                        gamma_a, gamma_scale, n, n_node_gammas_to_change, gamma_dist)
                elif p == 'equality':
                    probs_dist_dict[p] = equality_probs(
                        G.degree, n)
                else:
                    sys.exit('Probs string not recognized')

            probs = combine_probs(list(probs_dist_dict.values()))

            for _ in range(edge_increase_per_iteration):
                if nx.density(G) == 1:
                    return
                n1, n1_edges = get_node_1(G, probs)
                n2 = get_node_2(G, probs, n1_edges)
                G.add_edge(n1, n2)

                writer.writerow({'n1': n1, 'n2': n2, 'creation_time': i})

            if i % node_increase_on_iteration == 0:
                for _ in range(node_increase_per_iteration):
                    new_node = G.number_of_nodes()
                    G.add_node(new_node)


def get_node_1(G, probs):
    max_edges = len(probs)
    n1_node_list = list(probs.keys())
    n1_probs_list = list(probs.values())

    keep_going = True
    while keep_going:
        n1 = np.random.choice(n1_node_list, p=n1_probs_list)
        n1_edges = [nodes[1] for nodes in G.edges([n1])]
        n1_edges.append(n1)
        if len(n1_edges) < max_edges:
            keep_going = False
        else:
            index_n1 = n1_node_list.index(n1)
            del n1_node_list[index_n1]
            del n1_probs_list[index_n1]
            n1_probs_list = list(
                normalise(n1_probs_list).values())

    return n1, n1_edges


def get_node_2(G, probs, n1_edges):
    n2_node_list = list(probs.keys())
    n2_probs_list = list(probs.values())

    for node in n1_edges:
        index_node = n2_node_list.index(node)
        del n2_node_list[index_node]
        del n2_probs_list[index_node]

    n2_probs_list = list(normalise(n2_probs_list).values())

    return np.random.choice(n2_node_list, p=n2_probs_list)


def combine_probs(probs):
    combined_probs = []
    for prob in probs:
        norm_prob = normalise(prob)
        for i, p in norm_prob.items():
            if len(combined_probs) > i:
                combined_probs[i] *= p
            else:
                combined_probs.append(p)

    return normalise(combined_probs)


def equal_probs(n):
    e = {}
    for i in range(n):
        e[i] = 1 / n

    return e


def degree_probs(d, n, base_p):
    degree = dict(d)

    degree_probs = {}
    for i in range(n):
        if i in degree:
            degree_probs[i] = degree[i] + base_p
        else:
            degree_probs[i] = base_p

    return degree_probs


def gamma_probs(a, s, n):
    gamma_dist = stats.gamma.rvs(a, scale=s, size=n)

    return list(gamma_dist)


def static_gamma_probs(a, s, n, dist):
    gamma_dist = {}
    for i in range(n):
        if i not in dist:
            gamma_dist[i] = stats.gamma.rvs(a, scale=s)
        else:
            gamma_dist[i] = dist[i]

    return gamma_dist


def change_single_gamma_probs(a, s, node, dist):
    #     node = np.random.choice(list(dist.keys()))
    dist[node] = stats.gamma.rvs(a, scale=s)

    return dist


def static_gamma_with_individual_changes(a, s, n, t, dist):
    dist = static_gamma_probs(a, s, n, dist)
    for _ in range(t):
        dist = change_single_gamma_probs(
            a, s, np.random.choice(range(n)), dist)

    return dist


def equality_probs(degree, n):
    dp = normalise(degree_probs(degree, n, 1))
    ep = {}
    for i, p in dp.items():
        ep[i] = 1 - p
    return ep


def normalise(p):
    p_norm = {}
    if isinstance(p, list):
        sum_p = sum(p)
        for i in range(len(p)):
            p_norm[i] = p[i] / sum_p
    else:
        sum_p = sum(p.values())
        for i, p in p.items():
            p_norm[i] = p / sum_p

    return p_norm


for i in range(25):
    create_artificial_network(
        probs_list=['gamma_n_change', 'preferential_attachment'], iterations=5000, f_name_modifier=i)
# create_artificial_network(
#     probs_list=['preferential_attachment'], iterations=5000)
# create_artificial_network(probs_list=['fitness'], iterations=5000)
# create_artificial_network(probs_list=['gamma_n_change'], iterations=5000)
# create_artificial_network(
#     probs_list=['gamma_n_change', 'preferential_attachment'], iterations=5000)
# create_artificial_network(probs_list=['equality'], iterations=5000)
