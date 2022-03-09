from numpy.lib.type_check import nan_to_num
import math

from sklearn.metrics import r2_score
from sklearn.cluster import AgglomerativeClustering

import scipy.stats as stats
import numpy as np


def build_taxonomy_data_dict(plot_data_dict, add_net_stats=True):
    taxonomy_data_dict = {'mobility': {}, 'assortativity': {}, 'philanthropy': {},
                          'community': {}, 'delta_assortativity': {}, 'neighbourhood_mobility': {}}
    if add_net_stats:
        taxonomy_data_dict['equality'] = {}
        # taxonomy_data_dict['nodes'] = {}
        # taxonomy_data_dict['edges'] = {}

    for data_f_name, plot_data in plot_data_dict.items():

        taxonomy_data = plot_data['taxonomy_data']
        if add_net_stats:
            stats_data = plot_data['stats_data']
        t = plot_data['t']
        dt = plot_data['dt']
        dt_percent = int(plot_data['dt_percent'])
        data_type = plot_data['data_type']
        struc_type = plot_data['struc_type']

        data_label = f"{data_f_name}: t={t} dt={dt} #{data_type}${struc_type}"

        individual = taxonomy_data['individual']
        delta_individual = taxonomy_data['delta_individual']
        neighbourhood = taxonomy_data['neighbourhood']
        delta_neighbourhood = taxonomy_data['delta_neighbourhood']

        taxonomy_data_dict['mobility'][data_label], _ = nan_to_num(stats.pearsonr(
            individual, delta_individual))
        taxonomy_data_dict['assortativity'][data_label], _ = nan_to_num(stats.pearsonr(
            individual, neighbourhood))
        taxonomy_data_dict['philanthropy'][data_label], _ = nan_to_num(stats.pearsonr(
            individual, delta_neighbourhood))
        taxonomy_data_dict['community'][data_label], _ = nan_to_num(stats.pearsonr(
            delta_individual, neighbourhood))
        taxonomy_data_dict['delta_assortativity'][data_label], _ = nan_to_num(stats.pearsonr(
            delta_individual, delta_neighbourhood))
        taxonomy_data_dict['neighbourhood_mobility'][data_label], _ = nan_to_num(stats.pearsonr(
            neighbourhood, delta_neighbourhood))

        if add_net_stats:
            gini_dt = math.ceil(
                (len(stats_data['gini_coeff']) / 100) * dt_percent)
            taxonomy_data_dict['equality'][data_label] = (list(
                stats_data['gini_coeff'])[gini_dt] * 2) - 1

            # taxonomy_data_dict['nodes'][data_label] = (list(
            #     stats_data['nodes'])[gini_dt] / list(stats_data['nodes'])[-1] * 2) - 1
            # taxonomy_data_dict['edges'][data_label] = (list(
            #     stats_data['nodes'])[gini_dt] / list(stats_data['edges'])[-1] * 2) - 1

    return taxonomy_data_dict

# ---------------------------------------------------


def taxonomy_correlation_R2(taxonomy_data_dict):
    grid_correlations = []
    grid_r_square = []
    for x_label, x_data in taxonomy_data_dict.items():
        grid_c_line = []
        grid_r_line = []
        for y_label, y_data in taxonomy_data_dict.items():
            if y_label == x_label:
                grid_c_line.append(0)
                grid_r_line.append(0)
            else:
                correlation, _ = stats.pearsonr(
                    list(x_data.values()), list(y_data.values()))
                grid_c_line.append(correlation)

                grid_r_line.append(
                    r2_score(list(x_data.values()), list(y_data.values())))

        grid_correlations.append(grid_c_line)
        grid_r_square.append(grid_r_line)

    return grid_correlations, grid_r_square


def PCA(X, num_components, cov_mat=None):
    X_meaned = X - np.mean(X, axis=0)
    if not cov_mat:
        cov_mat = np.cov(X_meaned, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    X_reduced = np.dot(eigenvector_subset.transpose(),
                       X_meaned.transpose()).transpose()

    return X_reduced


def clustering(X, n_clusters):
    hc = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='euclidean', linkage='ward')

    y_hc = hc.fit_predict(X)

    return y_hc


# -------------------------------
