import matplotlib.pyplot as plt
from matplotlib import cm, colors, lines, patches


import scipy.stats as stats
import numpy as np
import datetime

import taxonomy_analysis
import taxonomy_plotting
import utilities


def plot_taxonomy_aspects_over_time(taxonomy_time_steps, clustering_type='data_type', highlighted_file=''):
    clustering_type_list, plot_colors = init_plot_temporal(
        taxonomy_time_steps, clustering_type)

    plot_data_dict = {}
    for timestep, taxonomy_data in taxonomy_time_steps.items():
        ts = 100 - int(timestep)
        taxonomy_data_dict = taxonomy_analysis.build_taxonomy_data_dict(
            taxonomy_data)

        for taxonomy_aspect, correlation_data in taxonomy_data_dict.items():
            for data_label, correlation in correlation_data.items():
                if clustering_type == 'data_type':
                    ll = data_label[data_label.index(
                        '#') + 1:data_label.index('$')]
                elif clustering_type == 'struc_type':
                    ll = data_label[data_label.index('$') + 1:]
                elif clustering_type == 'f_name':
                    ll = data_label[:data_label.index(':')]

                dl = f"{data_label[:data_label.index(':')]}:{ll}"

                if taxonomy_aspect in plot_data_dict:
                    if dl in plot_data_dict[taxonomy_aspect]:
                        plot_data_dict[taxonomy_aspect][dl][ts] = correlation
                    else:
                        plot_data_dict[taxonomy_aspect][dl] = {
                            ts: correlation}
                else:
                    plot_data_dict[taxonomy_aspect] = {
                        dl: {ts: correlation}}

    plot_name = f'{highlighted_file}{clustering_type}'
    plot_temporal(plot_data_dict, plot_colors, clustering_type_list,
                  'taxonomy_aspect_over_time', plot_name, 'Correlation', [-1.1, 1.1], highlighted_file)


def plot_low_degree_ratio(taxonomy_time_steps, clustering_type, degree_threshold=1, change_threshold=0, highlighted_file='IETF'):
    clustering_type_list, plot_colors = init_plot_temporal(
        taxonomy_time_steps, clustering_type)

    plotting_dict = {}
    for timestep, plot_data_dict in taxonomy_time_steps.items():
        ts = 100 - int(timestep)
        for data_f_name, plot_data in plot_data_dict.items():
            if clustering_type == 'data_type':
                ll = plot_data['data_type']
            elif clustering_type == 'struc_type':
                ll = plot_data['struc_type']
            elif clustering_type == 'f_name':
                ll = data_f_name

            dl = f"{data_f_name}:{ll}"

            taxonomy_df = plot_data['taxonomy_data']

            stats_dict = {
                'low_node_degree_ratio': taxonomy_df['individual'],
                'low_node_degree_change_ratio': taxonomy_df['delta_individual'],
                'low_neighbour_degree_ratio': taxonomy_df['neighbourhood'],
                'low_neighbour_degree_change_ratio': taxonomy_df['delta_neighbourhood']
            }

            for stat_name, stat in stats_dict.items():
                threshold = change_threshold if 'change' in stat_name else degree_threshold
                filtered_stat = [
                    d for d in stat if d <= threshold]

                stat_ratio = len(filtered_stat) / len(stat)
                if stat_name in plotting_dict:
                    if dl in plotting_dict[stat_name]:
                        plotting_dict[stat_name][dl][ts] = stat_ratio
                    else:
                        plotting_dict[stat_name][dl] = {
                            ts: stat_ratio}
                else:
                    plotting_dict[stat_name] = {
                        dl: {ts: stat_ratio}}

    plot_name = f'{highlighted_file}{clustering_type}'
    plot_temporal(plotting_dict, plot_colors, clustering_type_list,
                  "low_degree_ratio", plot_name, 'Ratio', [-0.1, 1.1], highlighted_file)


def plot_high_degree_nodes(taxonomy_time_steps, clustering_type, high_degree_node_count=100, highlighted_file='IETF'):
    clustering_type_list, plot_colors = init_plot_temporal(
        taxonomy_time_steps, clustering_type)

    plotting_dict = {}
    for timestep, plot_data_dict in taxonomy_time_steps.items():
        ts = 100 - int(timestep)
        for data_f_name, plot_data in plot_data_dict.items():
            print(data_f_name)
            if clustering_type == 'data_type':
                ll = plot_data['data_type']
            elif clustering_type == 'struc_type':
                ll = plot_data['struc_type']
            elif clustering_type == 'f_name':
                ll = data_f_name

            taxonomy_df = plot_data['taxonomy_data']

            stats_dict = {
                'high_node_degree_ratio': taxonomy_df['individual'],
                'high_node_degree_change_ratio': taxonomy_df['delta_individual'],
                'high_neighbour_degree_ratio': taxonomy_df['neighbourhood'],
                'high_neighbour_degree_change_ratio': taxonomy_df['delta_neighbourhood']
            }

            for stat_name, stat in stats_dict.items():
                print(stat_name)
                current_stat = list(stat)
                for i in range(high_degree_node_count):
                    dl = f"{data_f_name}_{i}:{ll}"

                    if current_stat:
                        current_max = max(current_stat)
                        current_stat.remove(current_max)

                    if stat_name in plotting_dict:
                        if dl in plotting_dict[stat_name]:
                            plotting_dict[stat_name][dl][ts] = current_max
                        else:
                            plotting_dict[stat_name][dl] = {
                                ts: current_max}
                    else:
                        plotting_dict[stat_name] = {
                            dl: {ts: current_max}}

    plot_name = f'{highlighted_file}{clustering_type}'
    plot_temporal(plotting_dict, plot_colors, clustering_type_list,
                  "high_degree_nodes", plot_name, 'Ratio', highlighted_file=highlighted_file)


#  --------------------------------


def init_plot_temporal(taxonomy_time_steps, clustering_type):
    timesteps = list(taxonomy_time_steps.keys())
    timesteps.sort()
    taxonomy_t0 = taxonomy_time_steps[timesteps[0]]
    if clustering_type == 'f_name':
        clustering_type_list = list(taxonomy_t0.keys())
    else:
        clustering_type_list = list(dict.fromkeys([pdd[clustering_type]
                                                   for pdd in list(taxonomy_t0.values())]))
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(clustering_type_list)), 'tab20')

    return(clustering_type_list, plot_colors)


def plot_temporal(plot_data_dict, plot_colors, clustering_type_list, plot_folder, plot_name, ylabel, ylim=None, highlighted_file=None, multi=False):
    legend_labels = []
    legend_elements = []

    if multi:
        _, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
    i = 0
    for plot_type, named_time_data in plot_data_dict.items():
        if multi and plot_type in ['equality']:
            # , 'assortativity', 'delta_assortativity']:
            ax = axes[i]
            continue
        else:
            _, ax = plt.subplots(1, 1, figsize=(15, 10))
        for data_name, plot_data in named_time_data.items():
            clt = data_name[data_name.index(':') + 1:]
            if highlighted_file and data_name.startswith(highlighted_file):
                type_colour = 'black'
                z_order = 3
            else:
                type_colour = plot_colors.to_rgba(
                    clustering_type_list.index(clt))
                z_order = 1
            legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(clt, legend_labels,
                                                                                      legend_elements, colour=type_colour)
            point_letter = \
                utilities.plot_letters[list(
                    utilities.dataset_type_lookup.keys()).index(data_name[:data_name.index(':')])] \
                if data_name[:data_name.index(':')] in list(utilities.dataset_type_lookup.keys()) else '.'

            x = list(plot_data.keys())
            x.sort()

            y = []
            for tsx in x:
                y.append(plot_data[tsx])

            ax.plot(x, y, color=type_colour, zorder=z_order)
            #  , linewidth=z_order)
            # ax.plot(x[0] - 1, y[0],
            #              marker=f"${point_letter}$", markersize=20, color=type_colour, zorder=z_order, linewidth=z_order)

        taxonomy_plotting.default_plot_params(ax, legend_elements)
        ax.set_title(plot_type, fontsize=utilities.plot_font_size)
        ax.set_xlabel('Percentage of Edges',
                      fontsize=utilities.plot_font_size)
        ax.set_ylabel(ylabel, fontsize=utilities.plot_font_size)
        if ylim:
            ax.set_ylim(ylim)

        i += 1

        plt.tight_layout()
        plt.savefig(
            f"./figs/{plot_folder}/{'multi_' if multi else ''}{plot_type}_{plot_name}.png")


# -------------------------


def plot_equality(plot_data_dict, y_type=None, clustering_type='data_type', use_neighbour=False):
    clustering_type_list = list(dict.fromkeys([pdd[clustering_type]
                                               for pdd in list(plot_data_dict.values())]))
    plot_colors = cm.ScalarMappable(colors.Normalize(
        0, len(clustering_type_list)), 'tab10')
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    legend_labels = []
    legend_elements = []
    for _, plot_data in plot_data_dict.items():
        stats_data = plot_data['stats_data']
        clt = plot_data[clustering_type]

        time_list = list(stats_data['creation_time'][1:])
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
            ax.set_xlabel('Normalised Time', fontsize=28)
        else:
            max_it = len(time_list)
            norm_time_list = [i / max_it for i, _ in enumerate(time_list)]
            ax.set_xlabel('Normalised Iteration',
                          fontsize=28)

        if use_neighbour:
            equality_list = list(stats_data['neighbour_gini_coeff'][1:])
        else:
            equality_list = list(stats_data['gini_coeff'][1:])

        type_colour = plot_colors.to_rgba(
            clustering_type_list.index(clt))
        legend_labels, legend_elements = taxonomy_plotting.custom_legend_elements(clt, legend_labels,
                                                                                  legend_elements, colour=type_colour)
        ax.plot(norm_time_list, equality_list, color=type_colour)
        ax.set_ylabel(
            f"{'neighbour_' if use_neighbour else ''}Gini Coefficient", fontsize=28)
        # ax.set_title('Equality over time')
        ax.set_ylim([-0.1, 1.1])

    taxonomy_plotting.default_plot_params(ax, legend_elements)

    plt.tight_layout()
    plt.savefig(
        f"./figs/{'neighbour_' if use_neighbour else ''}equality_comparison_{clustering_type}_{y_type}.png")
