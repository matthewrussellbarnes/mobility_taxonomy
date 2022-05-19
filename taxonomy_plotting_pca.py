import matplotlib.pyplot as plt
from matplotlib import cm, colors

import os
import numpy as np

import taxonomy_analysis
import taxonomy_plotting
import utilities


def plot_taxonomy_pca(plot_data_dict, dt_percent, pca_type='corr', clus_name_pair=None, n_cluster=6, corr_mat=None):

    corr_mat, clus_type_dict, plot_colors, taxonomy_data_per_dataset = init_plot_pca(
        plot_data_dict, clus_name_pair, pca_type, n_cluster)

    plot_pca(taxonomy_data_per_dataset, corr_mat, clus_type_dict, plot_colors,
             f'PCA{pca_type}_dtp{dt_percent}', 'taxonomy_pca')


def cluster_plot(points, n_cluster, plot_name, x_label='x', y_label='y'):
    points = np.array(points)
    cluster_mat = taxonomy_analysis.clustering(points, n_cluster)

    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, n_cluster), 'tab20')

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    for nc in range(n_cluster):
        ax.scatter(points[cluster_mat == nc, 0], points[cluster_mat ==
                                                        nc, 1], s=100, label=f"cluster{nc}", color=plot_colors.to_rgba(nc))

    ax.set_xlabel(x_label, fontsize=utilities.plot_font_size)
    ax.set_ylabel(y_label, fontsize=utilities.plot_font_size)
    ax.set_title('Clustering')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    taxonomy_plotting.default_plot_params(ax)

    plt.tight_layout()
    plt.savefig(
        f"./figs/clusters/cluster_{plot_name}_nc{n_cluster}.png")


def plot_taxonomy_pca_over_time(taxonomy_time_steps, pca_type='corr', clus_name_pair=None, n_cluster=6, highlighted_file=''):
    timesteps = list(taxonomy_time_steps.keys())
    timesteps.sort(reverse=True)
    taxonomy_t0 = taxonomy_time_steps[timesteps[0]]

    corr_mat, clus_type_dict, plot_colors, _ = init_plot_pca(
        taxonomy_t0, clus_name_pair, pca_type, n_cluster)

    taxonomy_data_per_timestep_dataset = {}
    for data_f_name in list(taxonomy_t0.keys()):
        taxonomy_timestep_data_dict = {}
        for timestep in timesteps:
            actual_timestep = 100 - int(timestep)
            taxonomy_timestep_data_dict[actual_timestep] = taxonomy_analysis.build_taxonomy_data_dict(
                {data_f_name: taxonomy_time_steps[timestep][data_f_name]})

        for timestep_l, timestep_data in taxonomy_timestep_data_dict.items():
            for _, aspect in timestep_data.items():
                for dataset, aspect_entry in aspect.items():
                    td_label = f'{timestep_l}_{dataset}'
                    if td_label in taxonomy_data_per_timestep_dataset:
                        taxonomy_data_per_timestep_dataset[td_label].append(
                            aspect_entry)
                    else:
                        taxonomy_data_per_timestep_dataset[td_label] = [
                            aspect_entry]

    plot_pca(taxonomy_data_per_timestep_dataset, corr_mat, clus_type_dict, plot_colors,
             f'{highlighted_file}PCA{pca_type}', 'multi_taxonomy_pca', highlighted_file=highlighted_file)


def init_plot_pca(plot_data_dict, clus_name_pair=None, pca_type='corr', n_cluster=6):
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
        f_name_list = [fn[:fn.index(':')]for fn in list(
            taxonomy_data_per_dataset.keys())]
        clus_type_dict['f_name'] = f_name_list

    if 'aggl' in clus_name_pair:
        clus_type_dict['aggl'] = list(range(n_cluster))

    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(clus_type_dict[clus_name_pair[0]])), 'tab20')

    return(corr_mat, clus_type_dict, plot_colors, taxonomy_data_per_dataset)


def plot_pca(pca_data_dict, corr_mat, clus_type_dict, plot_colors, plot_name, plot_folder, highlighted_file=''):
    # texts = []
    pca_taxonomy = taxonomy_analysis.PCA(
        list(pca_data_dict.values()), 2, corr_mat)

    d_label_list = list(pca_data_dict.keys())
    named_pca_taxonomy = {}
    for i in range(len(pca_taxonomy)):
        named_pca_taxonomy[d_label_list[i]] = list(pca_taxonomy[i])

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    legend_elements = []
    legend_labels = []
    if len(clus_type_dict) == 2:
        lc = {}
        lm = {}
        msd = {}
        for d_label, coor in named_pca_taxonomy.items():
            point_label = d_label[3:d_label.index(':')]
            point_data_dict = {}
            for cni, clus_name in enumerate(clus_type_dict.keys()):
                if 'data_type' == clus_name:
                    ll = d_label[d_label.index('#') + 1:d_label.index('$')]
                    ll_in = clus_type_dict[clus_name].index(ll)
                elif 'struc_type' == clus_name:
                    ll = d_label[d_label.index('$') + 1:]
                    ll_in = clus_type_dict[clus_name].index(ll)
                elif 'f_name' == clus_name:
                    ll = point_label
                    ll_in = clus_type_dict[clus_name].index(ll)
                elif 'aggl' == clus_name:
                    cluster_mat = taxonomy_analysis.clustering(
                        np.array(list(pca_data_dict.values())), len(clus_type_dict[clus_name]))
                    for nc in clus_type_dict[clus_name]:
                        points = np.array(
                            list(named_pca_taxonomy.values()))
                        for clus_coor in points[cluster_mat == nc]:
                            if clus_coor[0] == coor[0] and clus_coor[1] == coor[1]:
                                ll = f"Cluster {nc}"
                                ll_in = nc
                                break

                point_data_dict['l'] = \
                    utilities.plot_letters[list(
                        utilities.dataset_type_lookup.keys()).index(point_label)]

                if cni == 0:
                    if point_label == highlighted_file:
                        point_colour = 'black'
                    else:
                        point_colour = plot_colors.to_rgba(ll_in)
                    point_data_dict['c'] = point_colour
                    lc[ll] = point_colour
                elif cni == 1:
                    point_marker = utilities.plot_markers[ll_in]
                    point_data_dict['m'] = point_marker
                    lm[ll] = point_marker

            if point_label not in msd:
                point_data_dict['coorX'] = [coor[0]]
                point_data_dict['coorY'] = [coor[1]]
                msd[point_label] = point_data_dict
            else:
                msd[point_label]['coorX'].append(coor[0])
                msd[point_label]['coorY'].append(coor[1])

        for point_label, point_data in msd.items():
            coorX = point_data['coorX']
            coorY = point_data['coorY']
            ax.plot(coorX, coorY, color=point_data['c'])
            ax.scatter(coorX[1:], coorY[1:], marker=point_data['m'], edgecolors=point_data['c'],
                       facecolor='none', s=100)
            ax.plot(coorX[0], coorY[0], marker=f"${point_data['l']}$",
                    color=point_data['c'], markersize=20)
            # texts.append(ax.text(coorX[0], coorY[0], point_label))

        for ll_col, clus_col in lc.items():
            legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(
                ll_col, legend_labels, legend_elements, colour=clus_col)

        for ll_mar, clus_mar in lm.items():
            legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(
                ll_mar, legend_labels, legend_elements, marker=clus_mar)

    elif len(clus_type_dict) == 1:
        clus_name = list(clus_type_dict.keys())[0]
        if clus_name == 'aggl':
            cluster_mat = taxonomy_analysis.clustering(
                np.array(list(pca_data_dict.values())), len(clus_type_dict[clus_name]))
            for nc in clus_type_dict[clus_name]:
                points = np.array(list(named_pca_taxonomy.values()))
                ax.scatter(points[cluster_mat == nc, 0], points[cluster_mat ==
                                                                nc, 1], s=100, label=f"cluster{nc}", color=plot_colors.to_rgba(nc))
                # ax.plot(points[cluster_mat == nc, 0], points[cluster_mat ==
                #                                              nc, 1], label=f"cluster{nc}", color=plot_colors.to_rgba(nc))

        else:
            for d_label, coor in named_pca_taxonomy.items():
                if clus_name == 'data_type':
                    ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                        clus_type_dict[clus_name].index(d_label[d_label.index('#') + 1:d_label.index('$')])))
                    # ax.plot(coor[0], coor[1])
                elif clus_name == 'f_name':
                    ax.scatter(coor[0], coor[1], label=d_label, color=plot_colors.to_rgba(
                        clus_type_dict[clus_name].index(d_label[3:d_label.index(':')])))
                    # ax.plot(coor[0], coor[1])

    else:
        print('Too many or few clustering types chosen')
        sys.exit()

    # ax.set_xlabel('x', fontsize=utilities.plot_font_size)
    # ax.set_ylabel('y', fontsize=utilities.plot_font_size)
    # ax.set_title('PCA')
    # ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.27, 1.0])

    taxonomy_plotting.default_plot_params(ax, legend_elements)

    plt.tight_layout()
    # adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(
    # arrowstyle="->", color='r', lw=1))

    plot_name += f'_ct{list(clus_type_dict.keys())[0]}'

    if 'aggl' in clus_type_dict:
        plot_name += f"_nc{len(clus_type_dict['aggl'])}"
    if len(clus_type_dict) == 2:
        plot_name += f"_ct2{list(clus_type_dict.keys())[1]}"

    plt.savefig(os.path.join(utilities.figs_path,
                             plot_folder, f"{plot_name}.png"))
