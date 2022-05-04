import matplotlib.pyplot as plt
import os
import pandas as pd
import math

import MobilityTaxonomy
import taxonomy_plotting_pca
import taxonomy_plotting_temporal
import utilities

utilities.init()

taxonomy_time_steps = {}
for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        mt = MobilityTaxonomy.MobilityTaxonomy(
            file,
            utilities.dt_percent_list,
            os.path.basename(dirpath),
            utilities.structure_type_lookup[os.path.splitext(file)[0]])
        mt.build(save=True)

        data_f_name = os.path.splitext(file)[0]
        print(utilities.plot_letters[list(
            utilities.structure_type_lookup.keys()).index(data_f_name)],
            data_f_name, mt.data_type, mt.struc_type)

        for dt_percent, taxonomy_df in mt.taxonomy_df_dict.items():
            plot_data = {'taxonomy_data': taxonomy_df,
                         'stats_data': mt.stats_df,
                         't': mt.t,
                         'dt_percent': dt_percent,
                         'dt': int(math.ceil(int(mt.t) * (int(dt_percent) / 100))),
                         'data_type': mt.data_type,
                         'struc_type': mt.struc_type}

            if dt_percent in taxonomy_time_steps:
                taxonomy_time_steps[dt_percent][data_f_name] = plot_data

            else:
                taxonomy_time_steps[dt_percent] = {data_f_name: plot_data}

        plot_data[data_f_name] = taxonomy_time_steps

# pca_clus_type_list = [['data_type', 'aggl'], ['aggl'], ['data_type'],
        #   ['f_name'],
pca_clus_type_list = [['data_type', 'struc_type'], ['struc_type', 'aggl']]
for pct in pca_clus_type_list:
    print(pct)
    taxonomy_plotting_pca.plot_taxonomy_pca_over_time(
        taxonomy_time_steps, clus_name_pair=pct, highlighted_file='')

    taxonomy_plotting_temporal.plot_taxonomy_aspects_over_time(
        taxonomy_time_steps, clustering_type=pct[0], highlighted_file='')

    # taxonomy_plotting_temporal.plot_low_degree_ratio(
    #     taxonomy_time_steps, clustering_type=pct[0], highlighted_file='')

    # taxonomy_plotting_temporal.plot_high_degree_nodes(
    #     taxonomy_time_steps, clustering_type=pct[0], highlighted_file='')
