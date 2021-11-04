import os
import networkx as nx
import csv
import pandas as pd
import math
import datetime


import utilities
import compute_equality


def build_taxonomy(data_f_name, di_percent, first_row=0, max_rows=0):
    data_path = os.path.join(utilities.dataset_path, f"{data_f_name}.csv")

    stats_file_exists = os.path.exists(os.path.join(
        utilities.stats_data_path, f"{data_f_name}.txt"))

    G = nx.Graph()
    current_ct = 0
    ct_edges = {}
    with open(data_path, encoding='utf-8-sig') as csvfile:
        for max_i, _ in enumerate(csvfile):
            pass
        if max_rows < max_i:
            max_i = max_rows

    with open(data_path, encoding='utf-8-sig') as csvfile:
        t2 = max_i - 1
        t1 = t2 - int(math.ceil(max_i * (di_percent / 100)))
        print(t2, t1)

        stats_df = pd.DataFrame(
            columns=['creation_time', 'nodes', 'edges', 'gini_coeff'])
        reader = csv.DictReader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i < first_row:
                pass
            elif i > first_row + max_i:
                break
            else:
                n1 = row["n1"]
                n2 = row["n2"]
                creation_time = row["creation_time"]

                ct_edges[i] = [n1, n2]

                if current_ct == 0:
                    current_ct = creation_time
                elif current_ct != creation_time:
                    current_ct = creation_time
                    for edge in list(ct_edges.values()):
                        G.add_edge(edge[0], edge[1])

                    degree_dict = dict(G.degree)

                    if not stats_file_exists:
                        for ei in list(ct_edges.keys()):
                            if ei % math.ceil(t2 / 100) == 0:
                                stats_df.loc[len(stats_df.index)] = [
                                    creation_time, G.number_of_nodes(), G.number_of_edges(),
                                    compute_equality.gini_coefficient(degree_dict)]

                    if t1 in list(ct_edges.keys()):
                        t1_degree = degree_dict
                        t1_nei_degree = dict(nx.average_neighbor_degree(G))
                        G_t1 = G.copy()

                    elif t2 in list(ct_edges.keys()) or i >= t2:
                        taxonomy_df = pd.DataFrame(
                            columns=['node', 'individual', 'delta_individual',
                                     'neighbourhood', 'delta_neighbourhood'])

                        t2_degree = degree_dict

                        for node, t2_degree_node in t2_degree.items():
                            if G_t1.__contains__(node):
                                t1_degree_node = t1_degree[node]
                                t1_nei_degree_node = t1_nei_degree[node]

                                t2_nei_degree_node = 0
                                cons_nei = G_t1[node]
                                for nei_node in cons_nei:
                                    t2_nei_degree_node += G.degree[nei_node]

                                t2_nei_degree_node /= len(cons_nei)

                                taxonomy_df.loc[len(taxonomy_df.index)] = [
                                    node, t1_degree_node, t2_degree_node - t1_degree_node,
                                    t1_nei_degree_node, t2_nei_degree_node -
                                    t1_nei_degree_node]

                    ct_edges = {}

    create_taxonomy_data_file(
        f"{data_f_name}_dtp{di_percent}_ti{t2}", taxonomy_df)

    if not stats_file_exists:
        path = f"{utilities.stats_data_path}/{data_f_name}.txt"
        stats_df.to_csv(path, index=False)

    return taxonomy_df, t2


def create_taxonomy_data_file(f_name, taxonomy_df):
    taxonomy_data_f_path = utilities.get_file_path(
        f_name, utilities.taxonomy_data_path)

    if not taxonomy_data_f_path:
        path = f"{utilities.taxonomy_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        taxonomy_df.to_csv(path, index=False)
