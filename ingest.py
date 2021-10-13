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
        if max_rows > 0:
            max_i = max_rows
        else:
            for max_i, _ in enumerate(csvfile):
                pass

    with open(data_path, encoding='utf-8-sig') as csvfile:
        t = max_i - 1
        dt = t - int(math.ceil(max_i * (di_percent / 100)))
        print(t, dt)

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
                            if ei % math.ceil(t / 100) == 0:
                                stats_df.loc[len(stats_df.index)] = [
                                    creation_time, G.number_of_nodes(), G.number_of_edges(),
                                    compute_equality.gini_coeff(degree_dict)]

                    if dt in list(ct_edges.keys()):
                        t_degree = degree_dict
                        t_nei_degree = dict(nx.average_neighbor_degree(G))
                        G_t = G.copy()

                    elif t in list(ct_edges.keys()):
                        taxonomy_df = pd.DataFrame(
                            columns=['node', 'individual', 'delta_individual',
                                     'neighbourhood', 'delta_neighbourhood'])

                        dt_degree = degree_dict

                        for node, dt_degree_node in dt_degree.items():
                            if G_t.__contains__(node):
                                t_degree_node = t_degree[node]
                                t_nei_degree_node = t_nei_degree[node]

                                dt_nei_degree_node = 0
                                cons_nei = G_t[node]
                                for nei_node in cons_nei:
                                    dt_nei_degree_node += G.degree[nei_node]

                                dt_nei_degree_node /= len(cons_nei)

                                taxonomy_df.loc[len(taxonomy_df.index)] = [
                                    node, t_degree_node, dt_degree_node - t_degree_node,
                                    t_nei_degree_node, dt_nei_degree_node -
                                    t_nei_degree_node]

                    ct_edges = {}

    create_taxonomy_data_file(
        f"{data_f_name}_dtp{di_percent}_ti{t}", taxonomy_df)

    if not stats_file_exists:
        path = f"{utilities.stats_data_path}/{data_f_name}.txt"
        stats_df.to_csv(path, index=False)

    return taxonomy_df, t


def create_taxonomy_data_file(f_name, taxonomy_df):
    taxonomy_data_f_path = utilities.get_file_path(
        f_name, utilities.taxonomy_data_path)

    if not taxonomy_data_f_path:
        path = f"{utilities.taxonomy_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        taxonomy_df.to_csv(path, index=False)
