import matplotlib.pyplot as plt
import os
import math

import MobilityTaxonomy
import taxonomy_plotting
import taxonomy_plotting_pca
import utilities

utilities.init()

dt_percent_list = [utilities.dt_percent_list[0]]
mt_list = []
for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        mt = MobilityTaxonomy.MobilityTaxonomy(
            file,
            dt_percent_list,
            os.path.basename(dirpath),
            utilities.structure_type_lookup[os.path.splitext(file)[0]])
        mt.build(save=True)
        mt_list.append(mt)

for dt_percent in dt_percent_list:
    plot_data = {}
    for mob_tax in mt_list:
        # Plot bar chart of each mobility aspect correlation individually for each network in data corpus
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        taxonomy_plotting.plot_taxonomy_for_single_network(
            ax, mob_tax.taxonomy_df_dict[dt_percent], f"{mob_tax.networkf} with t={mob_tax.t} and dt={int(math.ceil(int(mob_tax.t) * (dt_percent / 100)))}")
        plt.savefig(
            f"./figs/taxomony_{mob_tax.networkf}_ti{mob_tax.t}_dti{int(math.ceil(int(mob_tax.t) * (dt_percent / 100)))}.png")

        plot_data[mob_tax.networkf] = {
            'taxonomy_data': mob_tax.taxonomy_df_dict[dt_percent],
            'stats_data': mob_tax.stats_df,
            't': mob_tax.t,
            'dt': int(math.ceil(int(mob_tax.t) * (dt_percent / 100))),
            'dt_percent': dt_percent,
            'data_type': mob_tax.data_type,
            'struc_type': mob_tax.struc_type
        }

    # Plot 2d principle component analysis for mobility taxonomy over all the data corpus
    pca_clus_type_list = [['data_type', 'aggl'], ['aggl'], ['data_type'],
                          ['f_name'], ['data_type', 'struc_type'], ['struc_type', 'aggl']]
    for pct in pca_clus_type_list:
        print(pct)
        taxonomy_plotting_pca.plot_taxonomy_pca(
            plot_data, dt_percent, clus_name_pair=pct)

    # Plot bar chart of each mobility aspect correlation over all the data corpus
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    taxonomy_plotting.plot_taxonomy_for_multiple_networks(
        ax, plot_data, dt_percent)

    # Plot mobility taxonomy correlations for all nodes over all the data corpus
    taxonomy_plotting.plot_taxonomy_for_each_network(plot_data, dt_percent)

    # Plot correlations between taxonomy aspects over all the data corpus
    taxonomy_plotting.plot_taxonomy_pairs_for_multiple_networks(
        plot_data, dt_percent)

    # Plot grid of correlation coefficents between taxonomy aspects over all the data corpus
    taxonomy_plotting.plot_grid_taxonomy_correlations(
        plot_data, dt_percent)
