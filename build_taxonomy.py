from networkx.exception import PowerIterationFailedConvergence
import pandas as pd
import networkx as nx


def build_taxonomy(network_df, t, delta_t):
    G = nx.Graph()

    network_df_groupby = network_df.groupby(by='creation_time')
    grouped_network_df = pd.DataFrame(network_df_groupby)

    if t > max(grouped_network_df.index):
        t = max(grouped_network_df.index)
    prev_t = t - delta_t

    timestamp = grouped_network_df.at[t, 0]
    prev_timestamp = grouped_network_df.at[prev_t, 0]

    for creation_time, index_list in network_df_groupby.groups.items():
        for index in index_list:
            row = network_df.loc[index].values
            G.add_edge(row[0], row[1])

        if creation_time == prev_timestamp:
            prev_t_individual = dict(G.degree)
            prev_t_neighbourhood = dict(nx.average_neighbor_degree(G))

        if creation_time == timestamp:
            taxonomy_df = pd.DataFrame(
                columns=['node', 'individual', 'delta_individual',
                         'neighbourhood', 'delta_neighbourhood'])
            t_individual = dict(G.degree)
            t_neighbourhood = dict(nx.average_neighbor_degree(G))

            for node, t_individual_deg in t_individual.items():
                t_neighbourhood_deg = t_neighbourhood[node]

                if node in prev_t_individual:
                    prev_t_individual_deg = prev_t_individual[node]
                    prev_t_neighbourhood_deg = prev_t_neighbourhood[node]
                else:
                    prev_t_individual_deg = 0
                    prev_t_neighbourhood_deg = 0

                taxonomy_df.loc[len(taxonomy_df.index)] = [
                    node, t_individual_deg, t_individual_deg - prev_t_individual_deg,
                    t_neighbourhood_deg, t_neighbourhood_deg - prev_t_neighbourhood_deg]

    return taxonomy_df
