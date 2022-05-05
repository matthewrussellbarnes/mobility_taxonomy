import os
import networkx as nx
import pandas as pd
import math
import datetime
import numpy as np

import utilities


class MobilityTaxonomy:
    def __init__(self, network_file_name, dt_percent_list, data_type, struc_type, max_rows=1000000):
        self.networkf = network_file_name
        self.data_type = data_type
        self.struc_type = struc_type
        self.dt_percent_list = dt_percent_list
        self.max_rows = max_rows

    def build(self, save=False):
        taxonomy_df_dict = {}
        data_f_name = os.path.splitext(self.networkf)[0]

        build_taxonomy_dt_list = []
        for dt_percent in self.dt_percent_list:
            taxonomy_data_f_path = utilities.get_file_path(
                f"{data_f_name}_dtp{dt_percent}", utilities.taxonomy_data_path)

            if taxonomy_data_f_path:
                taxonomy_df_dict[dt_percent] = pd.read_csv(
                    taxonomy_data_f_path)
                self.t = taxonomy_data_f_path[
                    taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2:
                    taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2)]
            else:
                build_taxonomy_dt_list.append(dt_percent)

        if build_taxonomy_dt_list:
            build_taxonomy_df_dict, self.stats_df, self.t = build_taxonomy(
                self.networkf, build_taxonomy_dt_list, max_rows=self.max_rows, save=save)
            taxonomy_df_dict = taxonomy_df_dict | build_taxonomy_df_dict
        else:
            stats_data_f_path = utilities.get_file_path(
                f"{data_f_name}", utilities.stats_data_path)
            self.stats_df = pd.read_csv(stats_data_f_path)

        self.taxonomy_df_dict = taxonomy_df_dict

#  ------------------------------------------


def build_taxonomy(networkf, di_percent_list, max_rows=0, save=True):
    data_path = os.path.join(utilities.dataset_path, networkf)

    data_f_name = os.path.splitext(networkf)[0]
    stats_file_exists = os.path.exists(os.path.join(
        utilities.stats_data_path, f"{data_f_name}.txt"))

    G = nx.Graph()

    t1_stats_dict = {}
    current_ct = 0
    with open(data_path, encoding='utf-8-sig') as csvfile:
        print("Opening", data_path)
        for max_i_file, _ in enumerate(csvfile):
            pass
        if max_rows < max_i_file and max_rows != 0:
            max_i = max_rows
        else:
            max_i = max_i_file

    with open(data_path, encoding='utf-8-sig') as csvfile:
        t2 = max_i - 1

        stats_df = pd.DataFrame(
            columns=['creation_time', 'nodes', 'edges', 'gini_coeff', 'neighbour_gini_coeff'])

        csvfile.readline()
        i = 0
        ct_list = []
        while(i < max_i):
            line = csvfile.readline().split()
            n1 = line[0]
            n2 = line[1]
            creation_time = line[2]
            ct_list.append(i)

            if G.has_edge(n1, n2) or n1 == n2:
                if max_i < max_i_file:
                    t2 += 1
                    max_i += 1
            else:
                G.add_edge(n1, n2)

            if current_ct == 0:
                current_ct = creation_time
            elif current_ct != creation_time or i >= t2:
                current_ct = creation_time

                if not stats_file_exists:
                    for ct_i in ct_list:
                        if ct_i % math.ceil(t2 / 100) == 0:
                            stats_df.loc[len(stats_df.index)] = [
                                creation_time, G.number_of_nodes(), G.number_of_edges(),
                                gini_coefficient(dict(G.degree)), gini_coefficient(nx.average_neighbor_degree(G))]

                for di_percent in di_percent_list:
                    t1_stats_dict = calculate_t1_stats(
                        G, t2, max_i, di_percent, ct_list, t1_stats_dict)

                if t2 in ct_list or i >= t2:
                    i = max_i
                ct_list = []
            i = i + 1

    taxonomy_df_dict = get_taxonomy_df_dict(G, t1_stats_dict)

    if save:
        for dipt, tdf in taxonomy_df_dict.items():
            create_taxonomy_data_file(
                f"{data_f_name}_dtp{dipt}_n{G.number_of_nodes()}_e{G.number_of_edges()}_di{t2}", tdf)

        if not stats_file_exists:
            path = f"{utilities.stats_data_path}/{data_f_name}.txt"
            stats_df.to_csv(path, index=False)

    return taxonomy_df_dict, stats_df, G.number_of_edges()


def calculate_t1_stats(G, t2, max_i, di_percent, ct_list, t1_stats_dict):
    di = t2 - int(math.ceil(max_i * (di_percent / 100)))
    if di in ct_list and not di_percent in t1_stats_dict:
        print("Added stat")
        t1_stats_dict[di_percent] = {
            'degree': dict(G.degree),
            'nei_degree': dict(nx.average_neighbor_degree(G)),
            'G': G.copy()
        }
    return t1_stats_dict


def get_taxonomy_df_dict(G, t1_stats_dict):
    print('t2 underway')
    t2_degree = dict(G.degree)

    taxonomy_df_dict = {}
    for dip, t1_stats in t1_stats_dict.items():
        node_list = []
        individual = []
        delta_individual = []
        neighbourhood = []
        delta_neighbourhood = []
        for node, t1_degree_node in t1_stats['degree'].items():
            t1_nei_degree_node = t1_stats['nei_degree'][node]

            t2_degree_node = t2_degree[node]
            t2_nei_degree_node = 0
            cons_nei = t1_stats['G'][node]
            for nei_node in cons_nei:
                t2_nei_degree_node += G.degree[nei_node]

            t2_nei_degree_node /= len(cons_nei)

            node_list.append(node)
            individual.append(t1_degree_node)
            delta_individual.append(t2_degree_node - t1_degree_node)
            neighbourhood.append(t1_nei_degree_node)
            delta_neighbourhood.append(t2_nei_degree_node - t1_nei_degree_node)

        taxonomy_df_dict[dip] = pd.DataFrame({'node': node_list, 'individual': individual, 'delta_individual': delta_individual,
                                              'neighbourhood': neighbourhood, 'delta_neighbourhood': delta_neighbourhood})
    return taxonomy_df_dict


def create_taxonomy_data_file(f_name, taxonomy_df):
    taxonomy_data_f_path = utilities.get_file_path(
        f_name, utilities.taxonomy_data_path)

    if not taxonomy_data_f_path:
        path = f"{utilities.taxonomy_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        taxonomy_df.to_csv(path, index=False)

#  ------------------------------------


def gini_coefficient(imp_dict):
    x = np.array(list(imp_dict.values()))
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
