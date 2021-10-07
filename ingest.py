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

    G = nx.Graph()
    current_ct = 0
    ct_edges = []
    with open(data_path, encoding='utf-8-sig') as csvfile:
        if max_rows > 0:
            max_i = max_rows
        else:
            for max_i, _ in enumerate(csvfile):
                pass

        t = max_i
        dt = int(math.ceil(max_i * (di_percent / 100)))

        equality_df = pd.DataFrame(columns=['iteration', 'gini_coeff'])
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

                if current_ct != creation_time:
                    current_ct = creation_time
                    for edge in ct_edges:
                        G.add_edge(edge[0], edge[1])

                    degree_dict = dict(G.degree)
                    gini = compute_equality.gini_coeff(degree_dict)

                    equality_df.loc[len(equality_df.index)] = [
                        creation_time, gini]

                    if i == t:
                        t_degree = degree_dict
                        t_equality = gini
                        t_nei_degree = dict(nx.average_neighbor_degree(G))
                        G_t = G.copy()

                    elif i == dt:
                        taxonomy_df = pd.DataFrame(
                            columns=['node', 'individual', 'delta_individual',
                                     'neighbourhood', 'delta_neighbourhood', 'equality', 'delta_equality'])

                        dt_degree = degree_dict
                        dt_equality = gini

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
                                    t_nei_degree_node, t_equality, dt_equality]

                    ct_edges = []
                else:
                    ct_edges.append([n1, n2])

    create_taxonomy_data_file(
        f"{data_f_name}_dtp{di_percent}_ti{t}", taxonomy_df)

    create_equality_data_file(
        f"{data_f_name}_dtp{di_percent}_ti{t}", equality_df)

    return taxonomy_df, t


def create_taxonomy_data_file(f_name, taxonomy_df):
    taxonomy_data_f_path = utilities.get_file_path(
        f_name, utilities.taxonomy_data_path)

    if not taxonomy_data_f_path:
        path = f"{utilities.taxonomy_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        taxonomy_df.to_csv(path, index=False)


def create_equality_data_file(f_name, equality_df):
    equality_data_f_path = utilities.get_file_path(
        f_name, utilities.equality_data_path)

    if not equality_data_f_path:
        path = f"{utilities.equality_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        equality_df.to_csv(path, index=False)
