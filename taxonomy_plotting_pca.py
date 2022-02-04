import matplotlib.pyplot as plt
from matplotlib import cm, colors

import numpy as np

import taxonomy_analysis
import taxonomy_plotting
import utilities


def plot_taxonomy_pca(plot_data_dict, dt_percent, ax=None, pca_type='corr', clus_name_pair=None, n_cluster=6, corr_mat=None):
    taxonomy_data_dict = taxonomy_analysis.build_taxonomy_data_dict(
        plot_data_dict)
    clus_type_dict = {}

    if 'data_type' in clus_name_pair:
        data_type_list = list(dict.fromkeys([pdd['data_type']
                                             for pdd in list(plot_data_dict.values())]))
        clus_type_dict['data_type'] = data_type_list

    if 'struc_type' in clus_name_pair:
        struc_type_list = list(dict.fromkeys([pdd['struc_type']
                                              for pdd in list(plot_data_dict.values())]))
        clus_type_dict['struc_type'] = struc_type_list

    if not corr_mat:
        if pca_type == 'corr':
            corr_mat, _ = taxonomy_analysis.taxonomy_correlation_R2(
                taxonomy_data_dict)
        else:
            corr_mat = None

    taxonomy_data_per_dataset = {}
    for aspect in list(taxonomy_data_dict.values()):
        for dataset, aspect_entry in aspect.items():
            if dataset in taxonomy_data_per_dataset:
                taxonomy_data_per_dataset[dataset].append(aspect_entry)
            else:
                taxonomy_data_per_dataset[dataset] = [aspect_entry]

    if 'f_name' in clus_name_pair:
        f_name_list = list(taxonomy_data_per_dataset.keys())
        clus_type_dict['f_name'] = f_name_list

    if 'aggl' in clus_name_pair:
        cluster_mat = taxonomy_analysis.clustering(
            np.array(list(taxonomy_data_per_dataset.values())), n_cluster)
        clus_type_dict['aggl'] = list(range(n_cluster))

    pca_taxonomy = PCA(
        list(taxonomy_data_per_dataset.values()), 2, corr_mat)

    named_pca_taxonomy = {}
    for i in range(len(pca_taxonomy)):
        named_pca_taxonomy[list(taxonomy_data_per_dataset.keys())
                           [i]] = list(pca_taxonomy[i])

    if ax:
        plot_folder = 'multi_taxonomy_pca'
        plot_name = f'PCA{pca_type}_ct{clus_name_pair[0]}'
    else:
        plot_folder = 'taxonomy_pca'
        plot_name = f'PCA{pca_type}_ct{clus_name_pair[0]}_dtp{dt_percent}'
        _, ax = plt.subplots(1, 1, figsize=(15, 10))

    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(clus_type_dict[clus_name_pair[0]])), 'tab10')

    legend_elements = []
    legend_labels = []
    if len(clus_name_pair) == 2:
        lc = {}
        lm = {}
        msd = {}
        for cni, clus_name in enumerate(clus_name_pair):
            for d_label, coor in named_pca_taxonomy.items():
                if d_label not in msd:
                    msd[d_label] = {
                        'coor': coor
                    }

                if 'data_type' == clus_name:
                    ll = d_label[d_label.index('#') + 1:d_label.index('$')]
                    ll_in = data_type_list.index(ll)
                elif 'struc_type' == clus_name:
                    ll = d_label[d_label.index('$') + 1:]
                    ll_in = struc_type_list.index(ll)
                elif 'aggl' == clus_name:
                    for nc in range(n_cluster):
                        points = np.array(
                            list(named_pca_taxonomy.values()))
                        for clus_coor in points[cluster_mat == nc]:
                            if clus_coor[0] == coor[0] and clus_coor[1] == coor[1]:
                                ll = f"Cluster {nc}"
                                ll_in = nc
                                break

                if cni == 0:
                    point_colour = plot_colors.to_rgba(ll_in)
                    msd[d_label]['c'] = point_colour
                    lc[ll] = point_colour
                elif cni == 1:
                    point_marker = utilities.plot_markers[ll_in]
                    msd[d_label]['m'] = point_marker
                    lm[ll] = point_marker

        for _, point_data in msd.items():
            coor = point_data['coor']
            # utilities.mscatter([coor[0]], [coor[1]], ax=ax, s=100, m=[
            # point_data['m']], color=point_data['c'])
            ax.plot(coor[0], coor[1], marker=point_data['m'],
                    color=point_data['c'])

        for ll_col, clus_col in lc.items():
            legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(
                ll_col, legend_labels, legend_elements, colour=clus_col)

        for ll_mar, clus_mar in lm.items():
            legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(
                ll_mar, legend_labels, legend_elements, marker=clus_mar)

    elif len(clus_name_pair) == 1:
        clustering_type = clus_name_pair[0]
        if clustering_type == 'aggl':
            for nc in range(n_cluster):
                points = np.array(list(named_pca_taxonomy.values()))
                # ax.scatter(points[cluster_mat == nc, 0], points[cluster_mat ==
                # nc, 1], s=100, label=f"cluster{nc}", color=plot_colors.to_rgba(nc))
                ax.plot(points[cluster_mat == nc, 0], points[cluster_mat ==
                                                             nc, 1], label=f"cluster{nc}", color=plot_colors.to_rgba(nc))

        else:
            for d_label, coor in named_pca_taxonomy.items():
                if clustering_type == 'data_type':
                    ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                        data_type_list.index(d_label[d_label.index('#') + 1:d_label.index('$')])))
                    ax.plot(coor[0], coor[1])
                elif clustering_type == 'f_name':
                    ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                        f_name_list.index(d_label)))
                    ax.plot(coor[0], coor[1])

    else:
        print('Too many or few clustering types chosen')
        sys.exit()

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_title('PCA')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    taxonomy_plotting.default_plot_params(ax, legend_elements)

    plt.tight_layout()

    if 'aggl' in clus_name_pair:
        plot_name += f"_nc{n_cluster}"
    if len(clus_name_pair) == 2:
        plot_name += f"_ct2{clus_name_pair[1]}"
    plt.savefig(
        f"./figs/{plot_folder}/{plot_name}.png")

    return(corr_mat)

    # for ncncnc in range(8):
    #     cluster_plot(list(named_pca_taxonomy.values()), ncncnc + 2,
    #                  plot_name)


def cluster_plot(points, n_cluster, plot_name, x_label='x', y_label='y'):
    points = np.array(points)
    cluster_mat = taxonomy_analysis.clustering(points, n_cluster)

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

    taxonomy_plotting.default_plot_params(ax)

    plt.tight_layout()
    plt.savefig(
        f"./figs/clusters/cluster_{plot_name}_nc{n_cluster}.png")


def plot_taxonomy_pca_over_time(taxonomy_time_steps, pca_type='corr', clus_name_pair=None, n_cluster=6):
    corr_mat = None
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    timesteps = list(taxonomy_time_steps.keys())
    timesteps.sort()
    for timestep in timesteps:
        print(timestep)
        plot_data_dict = taxonomy_time_steps[timestep]
        corr_mat = plot_taxonomy_pca(plot_data_dict, timestep, pca_type=pca_type, ax=ax,
                                     clus_name_pair=clus_name_pair, corr_mat=corr_mat, n_cluster=n_cluster)