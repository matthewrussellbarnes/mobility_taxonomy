import matplotlib.pyplot as plt
from matplotlib import cm, colors, lines, patches

from sklearn.metrics import r2_score
from sklearn.cluster import AgglomerativeClustering

import scipy.stats as stats
import numpy as np
import datetime

import utilities


def default_plot_params(ax, legend_elements=None):
    ax.minorticks_on()
    ax.grid(which="major", alpha=1)
    ax.grid(which="minor", alpha=0.2)
    ax.tick_params(axis='both', labelsize=15)
    if legend_elements:
        ax.legend(handles=legend_elements, bbox_to_anchor=(
            0.5, -0.2), loc='upper center', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)


def custom_legend_elements(label, legend_labels, legend_elements, colour=None, marker=None):
    if label not in legend_labels:
        legend_labels.append(label)
        if colour:
            legend_elements.insert(0, patches.Patch(facecolor=colour, edgecolor='w',
                                                    label=label))
        elif marker:
            legend_elements.append(lines.Line2D([0], [0], marker=marker, color='w', label=label[:-2],
                                                markerfacecolor='#000000', markersize=15))
    return legend_labels, legend_elements

# -----------------------


def build_taxonomy_data_dict(plot_data_dict, add_net_stats=True):
    taxonomy_data_dict = {'mobility': {}, 'assortativity': {}, 'philanthropy': {},
                          'community': {}, 'delta_assortativity': {}, 'neighbourhood_mobility': {}}
    if add_net_stats:
        taxonomy_data_dict['equality'] = {}
        # , 'nodes': {}, 'edges': {}}

    for data_f_name, plot_data in plot_data_dict.items():

        taxonomy_data = plot_data['taxonomy_data']
        if add_net_stats:
            stats_data = plot_data['stats_data']
        t = plot_data['t']
        dt = plot_data['dt']
        data_type = plot_data['data_type']

        data_label = f"{data_f_name}: t={t} dt={dt} #{data_type}"

        individual = taxonomy_data['individual']
        delta_individual = taxonomy_data['delta_individual']
        neighbourhood = taxonomy_data['neighbourhood']
        delta_neighbourhood = taxonomy_data['delta_neighbourhood']

        taxonomy_data_dict['mobility'][data_label], _ = stats.pearsonr(
            individual, delta_individual)
        taxonomy_data_dict['assortativity'][data_label], _ = stats.pearsonr(
            individual, neighbourhood)
        taxonomy_data_dict['philanthropy'][data_label], _ = stats.pearsonr(
            individual, delta_neighbourhood)
        taxonomy_data_dict['community'][data_label], _ = stats.pearsonr(
            delta_individual, neighbourhood)
        taxonomy_data_dict['delta_assortativity'][data_label], _ = stats.pearsonr(
            delta_individual, delta_neighbourhood)
        taxonomy_data_dict['neighbourhood_mobility'][data_label], _ = stats.pearsonr(
            neighbourhood, delta_neighbourhood)

        if add_net_stats:
            taxonomy_data_dict['equality'][data_label] = (list(
                stats_data['gini_coeff'])[-1] + 1) / 2
        # taxonomy_data_dict['nodes'][data_label] = list(stats_data['nodes'])[-1]
        # taxonomy_data_dict['edges'][data_label] = list(stats_data['edges'])[-1]

    return taxonomy_data_dict


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

def plot_mobility(ax, taxonomy_data, color='black', curve_label=''):
    individual = taxonomy_data['individual']
    delta_individual = taxonomy_data['delta_individual']

    ax.scatter(individual, delta_individual, label=curve_label,
               color=color)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Change in Individual Degree")
    ax.set_title('Individual Mobility')

    default_plot_params(ax)


def plot_neighbourhood_mobility(ax, taxonomy_data, color='black', curve_label=''):
    neighbourhood = taxonomy_data['neighbourhood']
    delta_neighbourhood = taxonomy_data['delta_consistent_neighbourhood']

    ax.scatter(neighbourhood, delta_neighbourhood,
               label=curve_label, color=color)
    ax.set_xlabel("Neighbourhood Degree")
    ax.set_ylabel("Change in Neighbourhood Degree")
    ax.set_title('Neighbourhood Mobility')

    default_plot_params(ax)


def plot_assortativity(ax, taxonomy_data, color='black', curve_label=''):
    individual = taxonomy_data['individual']
    neighbourhood = taxonomy_data['neighbourhood']

    ax.scatter(individual, neighbourhood, label=curve_label,
               color=color)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Average Neighbourhood Degree")
    ax.set_title('Assortativity')

    default_plot_params(ax)


def plot_delta_assortativity(ax, taxonomy_data, color='black', curve_label=''):
    delta_individual = taxonomy_data['delta_individual']
    delta_neighbourhood = taxonomy_data['delta_consistent_neighbourhood']

    ax.scatter(delta_individual, delta_neighbourhood,
               label=curve_label, color=color)
    ax.set_xlabel("Change in Individual Degree")
    ax.set_ylabel("Change in Average Neighbourhood Degree")
    ax.set_title('Change in Assortativity')

    default_plot_params(ax)


def plot_philanthropy(ax, taxonomy_data, color='black', curve_label=''):
    individual = taxonomy_data['individual']
    delta_neighbourhood = taxonomy_data['delta_consistent_neighbourhood']

    ax.scatter(individual, delta_neighbourhood, label=curve_label,
               color=color)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Change in Average Neighbourhood Degree")
    ax.set_title('Philanthropy')

    default_plot_params(ax)


def plot_individuality_vs_community(ax, taxonomy_data, color='black', curve_label=''):
    delta_individual = taxonomy_data['delta_individual']
    neighbourhood = taxonomy_data['neighbourhood']

    ax.scatter(delta_individual, neighbourhood, label=curve_label,
               color=color)
    ax.set_xlabel("Change in Individual Degree")
    ax.set_ylabel("Average Neighbourhood Degree")
    ax.set_title('Individuality/Community')

    default_plot_params(ax)

# -----------------------------


def plot_taxonomy_for_single_network(ax, taxonomy_data, plot_label=''):
    individual = taxonomy_data['individual']
    delta_individual = taxonomy_data['delta_individual']
    neighbourhood = taxonomy_data['neighbourhood']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    mobility, _ = stats.pearsonr(individual, delta_individual)
    assortativity, _ = stats.pearsonr(individual, neighbourhood)
    philanthropy, _ = stats.pearsonr(individual, delta_neighbourhood)
    individuality_community, _ = stats.pearsonr(
        delta_individual, neighbourhood)
    delta_assortativity, _ = stats.pearsonr(
        delta_individual, delta_neighbourhood)
    neighbourhood_mobility, _ = stats.pearsonr(
        neighbourhood, delta_neighbourhood)

    ax.bar(['mobility', 'assortativity', 'philanthropy', 'commmunity', 'delta_assortativity', 'nbrhd_mobility'], [
           mobility, assortativity, philanthropy, individuality_community, delta_assortativity, neighbourhood_mobility])

    ax.set_xlabel('Taxonomy', fontsize=15)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=15)
    ax.set_title(f"Taxonomy for {plot_label}")
    ax.set_ylim([-1.1, 1.1])

    default_plot_params(ax)


def plot_taxonomy_for_multiple_networks(ax, plot_data_dict, dt_percent):
    x_axis_labels = ['mobility', 'assortativity', 'philanthropy',
                     'commmunity', 'delta_assortativity', 'nbrhd_mobility']
    r = np.arange(len(x_axis_labels))
    width = 0.03

    data_type_list = list(dict.fromkeys([pdd['data_type']
                                         for pdd in list(plot_data_dict.values())]))
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(data_type_list)), 'tab20')

    td_i = 0
    even = True
    for data_f_name, plot_data in plot_data_dict.items():

        taxonomy_data = plot_data['taxonomy_data']
        t = plot_data['t']
        dt = plot_data['dt']
        data_type = plot_data['data_type']

        individual = taxonomy_data['individual']
        delta_individual = taxonomy_data['delta_individual']
        neighbourhood = taxonomy_data['neighbourhood']
        # delta_neighbourhood = taxonomy_data['delta_neighbourhood']
        delta_consistent_neighbourhood = taxonomy_data['delta_consistent_neighbourhood']

        mobility, _ = stats.pearsonr(individual, delta_individual)
        assortativity, _ = stats.pearsonr(individual, neighbourhood)
        # philanthropy, _ = stats.pearsonr(individual, delta_neighbourhood)
        philanthropy_con, _ = stats.pearsonr(
            individual, delta_consistent_neighbourhood)
        individuality_community, _ = stats.pearsonr(
            delta_individual, neighbourhood)
        # delta_assortativity, _ = stats.pearsonr(
        # delta_individual, delta_neighbourhood)
        delta_assortativity_con, _ = stats.pearsonr(
            delta_individual, delta_consistent_neighbourhood)
        # neighbourhood_mobility, _ = stats.pearsonr(
        # neighbourhood, delta_neighbourhood)
        neighbourhood_mobility_con, _ = stats.pearsonr(
            neighbourhood, delta_consistent_neighbourhood)

        curve_label = f"{data_f_name}: t={t} dt={dt} {data_type}"

        ax.bar(r + (width * td_i), [
            mobility, assortativity, philanthropy_con, individuality_community, delta_assortativity_con, neighbourhood_mobility_con],
            width=width, label=curve_label, color=plot_colors.to_rgba(data_type_list.index(data_type)))

        if even:
            td_i = abs(td_i)
            td_i += 1
            even = False
        else:
            td_i *= -1
            even = True

    plt.xticks(r + width / len(plot_data_dict), x_axis_labels)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(25)

    ax.set_xlabel('Taxonomy', fontsize=15)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=15)
    ax.set_title(f"Taxonomy Comparison")
    ax.set_ylim([-1.1, 1.1])

    default_plot_params(ax)

    plt.tight_layout()
    plt.savefig(f"./figs/taxomony_type_comparison_all_dtp{dt_percent}.png")


def plot_taxonomy_for_each_network(plot_data_dict, dt_percent):
    fig1, ax_mobility = plt.subplots(1, 1, figsize=(15, 10))
    fig2, ax_assortativity = plt.subplots(1, 1, figsize=(15, 10))
    fig3, ax_philanthropy = plt.subplots(1, 1, figsize=(15, 10))
    fig4, ax_individuality_vs_community = plt.subplots(1, 1, figsize=(15, 10))
    fig5, ax_delta_assortativity = plt.subplots(1, 1, figsize=(15, 10))
    fig6, ax_neighbourhood_mobility = plt.subplots(1, 1, figsize=(15, 10))

    data_f_name_list = list(plot_data_dict.keys())
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(data_f_name_list)), 'tab20')

    for data_f_name, plot_data in plot_data_dict.items():

        taxonomy_data = plot_data['taxonomy_data']
        t = plot_data['t']
        dt = plot_data['dt']

        curve_label = f"{data_f_name}: t={t} dt={dt}"
        plot_color = plot_colors.to_rgba(data_f_name_list.index(data_f_name))

        plot_mobility(ax_mobility, taxonomy_data,
                      plot_color, curve_label)
        plot_assortativity(ax_assortativity, taxonomy_data,
                           plot_color, curve_label)
        plot_philanthropy(ax_philanthropy, taxonomy_data,
                          plot_color, curve_label)
        plot_individuality_vs_community(
            ax_individuality_vs_community, taxonomy_data, plot_color, curve_label)
        plot_delta_assortativity(ax_delta_assortativity,
                                 taxonomy_data, plot_color, curve_label)
        plot_neighbourhood_mobility(
            ax_neighbourhood_mobility, taxonomy_data, plot_color, curve_label)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()

    fig1.savefig(f"./figs/mobility_comparison_all_dtp{dt_percent}.png")
    fig2.savefig(f"./figs/assortativity_comparison_all_dtp{dt_percent}.png")
    fig3.savefig(f"./figs/philanthropy_comparison_all_dtp{dt_percent}.png")
    fig4.savefig(
        f"./figs/individuality_vs_community_comparison_all_dtp{dt_percent}.png")
    fig5.savefig(
        f"./figs/delta_assortativity_comparison_all_dtp{dt_percent}.png")
    fig6.savefig(
        f"./figs/neighbourhood_mobility_comparison_all_dtp{dt_percent}.png")


def plot_taxonomy_pairs_for_multiple_networks(plot_data_dict, dt_percent):
    taxonomy_data_dict = build_taxonomy_data_dict(plot_data_dict, False)

    data_type_list = list(dict.fromkeys([pdd['data_type']
                                         for pdd in list(plot_data_dict.values())]))
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(data_type_list)), 'tab10')

    legend_labels = []
    legend_elements = []
    used_taxonomies = []
    for x_label, x_data in taxonomy_data_dict.items():
        used_taxonomies.append(x_label)
        filtered_tdd = dict(
            filter(lambda x: x[0] not in used_taxonomies, taxonomy_data_dict.items()))
        for y_label, y_data in filtered_tdd.items():
            _, ax = plt.subplots(1, 1, figsize=(15, 10))
            for d_label, x_corr in x_data.items():
                y_corr = y_data[d_label]

                type_colour = plot_colors.to_rgba(
                    data_type_list.index(d_label[d_label.index('#') + 1:]))
                legend_labels, legend_elements = custom_legend_elements(
                    d_label[d_label.index('#') + 1:], legend_labels, legend_elements, colour=type_colour)
                ax.scatter(x_corr, y_corr, color=type_colour)
            ax.set_xlabel(x_label, fontsize=15)
            ax.set_ylabel(y_label, fontsize=15)
            ax.set_title(f"{x_label} vs {y_label} Correlation Comparison")
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])

            default_plot_params(ax, legend_elements)

            plt.tight_layout()
            plt.savefig(
                f"./figs/pair_taxonomy_type_comparison/{x_label}_{y_label}_comparison_dtp{dt_percent}.png")


def plot_grid_taxonomy_correlations(plot_data_dict, dt_percent):
    taxonomy_data_dict = build_taxonomy_data_dict(plot_data_dict, False)
    plot_labels = [label.replace('_', ' ').title().replace('Delta', 'Change in')
                   for label, _ in taxonomy_data_dict.items()]

    grid_correlations, grid_r_square = taxonomy_correlation_R2(
        taxonomy_data_dict)

    for plot_name, grid_data in {'correlation': grid_correlations, 'R2': grid_r_square}.items():
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        if plot_name == 'correlation':
            im = ax.imshow(grid_data, cmap='bwr')
            im.set_clim(-1, 1)
        else:
            im = ax.imshow(grid_data, cmap='Wistia')

        ax.set_title(f"Taxonomy {plot_name} Grid {dt_percent}")

        for pos_x in range(len(grid_data)):
            for pos_y in range(len(grid_data[pos_x])):
                label = round(grid_data[pos_x][pos_y], 2)
                ax.text(pos_x, pos_y, label, color='black',
                        ha='center', va='center', fontsize=15)

        ax.set_xticklabels(plot_labels)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(plot_labels)
        ax.set_yticks([0, 1, 2, 3, 4, 5])

        ax.tick_params(axis='both', labelsize=15)
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(25)

        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig(
            f"./figs/taxonomy_grid/grid_{plot_name}_dt{dt_percent}.png")

# -------------------


def plot_taxonomy_pca(plot_data_dict, dt_percent, pca_type='corr', clustering_type=None, n_cluster=6, clustering_type2=None):
    taxonomy_data_dict = build_taxonomy_data_dict(plot_data_dict)

    if clustering_type == 'data_type' or clustering_type2 == 'data_type':
        data_type_list = list(dict.fromkeys([pdd['data_type']
                                             for pdd in list(plot_data_dict.values())]))
        plot_colors = cm.ScalarMappable(colors.Normalize(
            0, len(data_type_list)), 'tab10')

    if pca_type == 'corr':
        corr_mat, _ = taxonomy_correlation_R2(taxonomy_data_dict)
    else:
        corr_mat = None

    taxonomy_data_per_dataset = {}
    for aspect in list(taxonomy_data_dict.values()):
        for dataset, aspect_entry in aspect.items():
            if dataset in taxonomy_data_per_dataset:
                taxonomy_data_per_dataset[dataset].append(aspect_entry)
            else:
                taxonomy_data_per_dataset[dataset] = [aspect_entry]

    if clustering_type == 'f_name':
        data_f_name_list = list(taxonomy_data_per_dataset.keys())
        plot_colors = cm.ScalarMappable(colors.Normalize(
            0, len(data_f_name_list)), 'hsv')
    elif clustering_type2:
        if clustering_type2 == 'aggl' or clustering_type == 'aggl':
            cluster_mat = clustering(
                np.array(list(taxonomy_data_per_dataset.values())), n_cluster)
    elif clustering_type == 'aggl':
        cluster_mat = clustering(
            np.array(list(taxonomy_data_per_dataset.values())), n_cluster)
        plot_colors = cm.ScalarMappable(colors.Normalize(
            0, n_cluster), 'tab10')

    pca_taxonomy = PCA(list(taxonomy_data_per_dataset.values()), 2, corr_mat)

    named_pca_taxonomy = {}
    for i in range(len(pca_taxonomy)):
        named_pca_taxonomy[list(taxonomy_data_per_dataset.keys())
                           [i]] = list(pca_taxonomy[i])

    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    legend_elements = []
    if clustering_type2:
        legend_labels = []
        for d_label, coor in named_pca_taxonomy.items():
            for nc in range(n_cluster):
                points = np.array(list(named_pca_taxonomy.values()))
                for clus_coor in points[cluster_mat == nc]:
                    if clus_coor[0] == coor[0] and clus_coor[1] == coor[1]:
                        ll_clus = f"Cluster {nc}"
                        clus_marker = utilities.plot_markers[nc]
                        legend_labels, legend_elements = custom_legend_elements(
                            ll_clus, legend_labels, legend_elements, marker=clus_marker)

                        ll_col = d_label[d_label.index('#') + 1:]
                        type_colour = plot_colors.to_rgba(
                            data_type_list.index(ll_col))
                        legend_labels, legend_elements = custom_legend_elements(
                            ll_col, legend_labels, legend_elements, colour=type_colour)

                        utilities.mscatter([coor[0]], [coor[1]], ax=ax, s=100, m=[
                                           clus_marker], color=type_colour)

    else:
        for d_label, coor in named_pca_taxonomy.items():
            if clustering_type == 'data_type':
                ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                    data_type_list.index(d_label[d_label.index('#') + 1:])))
            elif clustering_type == 'f_name':
                ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                    data_f_name_list.index(d_label)))

        if clustering_type == 'aggl':
            for nc in range(n_cluster):
                points = np.array(list(named_pca_taxonomy.values()))
                ax.scatter(points[cluster_mat == nc, 0], points[cluster_mat ==
                                                                nc, 1], s=100, label=f"cluster{nc}", color=plot_colors.to_rgba(nc))

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_title('PCA')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    default_plot_params(ax, legend_elements)

    plt.tight_layout()
    plot_name = f'PCA{pca_type}_ct{clustering_type}_dtp{dt_percent}'
    if clustering_type == 'aggl' or clustering_type2 == 'aggl':
        plot_name += f"_nc{n_cluster}"
    if clustering_type2:
        plot_name += f"_ct2{clustering_type2}"
    plt.savefig(
        f"./figs/taxonomy_pca/{plot_name}.png")

    # for ncncnc in range(8):
    #     cluster_plot(list(named_pca_taxonomy.values()), ncncnc + 2,
    #                  plot_name)


def cluster_plot(points, n_cluster, plot_name, x_label='x', y_label='y'):
    points = np.array(points)
    cluster_mat = clustering(points, n_cluster)

    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, n_cluster), 'tab20')

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    for nc in range(n_cluster):
        ax.scatter(points[cluster_mat == nc, 0], points[cluster_mat ==
                                                        nc, 1], s=100, label=f"cluster{nc}", color=plot_colors.to_rgba(nc))

    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title('Clustering')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    default_plot_params(ax)

    plt.tight_layout()
    plt.savefig(
        f"./figs/clusters/cluster_{plot_name}_nc{n_cluster}.png")


def plot_equality(plot_data_dict, y_type=None):
    data_type_list = list(dict.fromkeys([pdd['data_type']
                                         for pdd in list(plot_data_dict.values())]))
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(data_type_list)), 'tab10')
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    legend_labels = []
    legend_elements = []
    for _, plot_data in plot_data_dict.items():
        stats_data = plot_data['stats_data']
        data_type = plot_data['data_type']

        time_list = list(stats_data['creation_time'])
        if y_type == 'norm_time':
            max_time = max(time_list)
            min_time = min(time_list)
            norm_time_list = []
            for t in time_list:
                if '-' in str(t):
                    date_format = "%Y-%m-%d"

                    unix_t = datetime.datetime.timestamp(
                        datetime.datetime.strptime(t, date_format))
                    unix_max_time = datetime.datetime.timestamp(
                        datetime.datetime.strptime(max_time, date_format))
                    unix_min_time = datetime.datetime.timestamp(
                        datetime.datetime.strptime(min_time, date_format))

                    norm_time_list.append(
                        (unix_t - unix_min_time) / (unix_max_time - unix_min_time))
                else:
                    norm_time_list.append(
                        (t - min_time) / (max_time - min_time))
            ax.set_xlabel('Normalised Time', fontsize=15)
        else:
            max_it = len(time_list)
            norm_time_list = [i / max_it for i, _ in enumerate(time_list)]
            ax.set_xlabel('Normalised Iteration', fontsize=15)

        equality_list = list(stats_data['gini_coeff'])

        type_colour = plot_colors.to_rgba(
            data_type_list.index(data_type))
        legend_labels, legend_elements = custom_legend_elements(data_type, legend_labels,
                                                                legend_elements, colour=type_colour)
        ax.plot(norm_time_list, equality_list, color=type_colour)
        ax.set_ylabel('Equality', fontsize=15)
        ax.set_title('Equality over time')
        ax.set_ylim([-0.1, 1.1])

    default_plot_params(ax, legend_elements)

    plt.tight_layout()
    plt.savefig(
        f"./figs/equality_comparison_data_type_{y_type}.png")
