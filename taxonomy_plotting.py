import matplotlib.pyplot as plt
from matplotlib import cm, colors, lines, patches


import scipy.stats as stats
import numpy as np
import os

import taxonomy_analysis
import utilities


def default_plot_params(ax, legend_elements=None):
    ax.minorticks_on()
    ax.grid(which="major", alpha=1)
    ax.grid(which="minor", alpha=0.2)
    ax.tick_params(axis='both', labelsize=30)
    if legend_elements:
        ax.legend(handles=legend_elements,
                  # bbox_to_anchor=(0.5, -0.2), loc='upper center',
                  ncol=2)
    else:
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)


def custom_legend_elements(label, legend_labels, legend_elements, colour=None, marker=None):
    if label not in legend_labels:
        legend_labels.append(label)
        if colour:
            legend_elements.insert(0, patches.Patch(facecolor=colour, edgecolor='w',
                                                    label=label))
        elif marker:
            legend_elements.append(lines.Line2D([0], [0], marker=marker, color='w', label=label,
                                                markerfacecolor='#000000', markersize=15))
    return legend_labels, legend_elements

# -----------------------


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
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

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
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    ax.scatter(delta_individual, delta_neighbourhood,
               label=curve_label, color=color)
    ax.set_xlabel("Change in Individual Degree")
    ax.set_ylabel("Change in Average Neighbourhood Degree")
    ax.set_title('Change in Assortativity')

    default_plot_params(ax)


def plot_philanthropy(ax, taxonomy_data, color='black', curve_label=''):
    individual = taxonomy_data['individual']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

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

    ax.set_xlabel('Taxonomy', fontsize=utilities.plot_font_size)
    ax.set_ylabel('Pearson Correlation Coefficient',
                  fontsize=utilities.plot_font_size)
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

        curve_label = f"{data_f_name}: t={t} dt={dt} {data_type}"

        ax.bar(r + (width * td_i), [
            mobility, assortativity, philanthropy, individuality_community, delta_assortativity, neighbourhood_mobility],
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
        tick.set_pad(utilities.plot_font_size)

    ax.set_xlabel('Taxonomy', fontsize=utilities.plot_font_size)
    ax.set_ylabel('Pearson Correlation Coefficient',
                  fontsize=utilities.plot_font_size)
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
    taxonomy_data_dict = taxonomy_analysis.build_taxonomy_data_dict(
        plot_data_dict, False)

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
                data_type = d_label[d_label.index('#') + 1:d_label.index('$')]
                type_colour = plot_colors.to_rgba(
                    data_type_list.index(data_type))
                legend_labels, legend_elements = custom_legend_elements(
                    data_type, legend_labels, legend_elements, colour=type_colour)
                ax.scatter(x_corr, y_corr, color=type_colour)
            ax.set_xlabel(x_label, fontsize=utilities.plot_font_size)
            ax.set_ylabel(y_label, fontsize=utilities.plot_font_size)
            ax.set_title(f"{x_label} vs {y_label} Correlation Comparison")
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])

            default_plot_params(ax, legend_elements)

            plt.tight_layout()
            plt.savefig(
                f"./figs/pair_taxonomy_type_comparison/{x_label}_{y_label}_comparison_dtp{dt_percent}.png")

# --------------------------------


def plot_grid_taxonomy_correlations(plot_data_dict, dt_percent, with_gini=True):
    taxonomy_data_dict = taxonomy_analysis.build_taxonomy_data_dict(
        plot_data_dict, with_gini)
    plot_labels = [label.replace('_', ' ').title().replace('Delta', 'Change in')
                   for label, _ in taxonomy_data_dict.items()]

    grid_correlations, grid_r_square = taxonomy_analysis.taxonomy_correlation_R2(
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
                if str(label) == '0':
                    label = '-'
                ax.text(pos_x, pos_y, label, color='black',
                        ha='center', va='center', fontsize=utilities.plot_font_size)

        ax.set_xticklabels(plot_labels)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_yticklabels(plot_labels)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])

        ax.tick_params(axis='both', labelsize=15)
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(utilities.plot_font_size)

        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(utilities.figs_path, "taxonomy_grid",
                                 f"grid_{plot_name}_dt{dt_percent}{'_with_gini'if with_gini else ''}.png"))
