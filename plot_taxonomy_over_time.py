import matplotlib.pyplot as plt
import os
import pandas as pd
import math

import taxonomy_plotting_pca
import utilities

utilities.init()

taxonomy_time_steps = {}
for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    print(list(files))
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)
        data_type = os.path.basename(dirpath)
        struc_type = utilities.structure_type_lookup[data_f_name]

        taxonomy_data_f_path_list = utilities.get_file_path_for_multiple(
            f"{data_f_name}", utilities.taxonomy_data_path)
        stats_data_f_path = utilities.get_file_path(
            f"{data_f_name}", utilities.stats_data_path)

        stats_df = pd.read_csv(stats_data_f_path)
        for taxonomy_data_f_path in taxonomy_data_f_path_list:
            t = taxonomy_data_f_path[
                taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2:
                taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2)]
            dt_percent = taxonomy_data_f_path[
                taxonomy_data_f_path.index("_dtp") + 4:
                taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_dtp") + 4)]
            taxonomy_df = pd.read_csv(taxonomy_data_f_path)

            plot_data = {'taxonomy_data': taxonomy_df,
                         'stats_data': stats_df,
                         't': t,
                         'dt': int(math.ceil(int(t) * (int(dt_percent) / 100))),
                         'data_type': data_type,
                         'struc_type': struc_type}
            if dt_percent in taxonomy_time_steps:
                taxonomy_time_steps[dt_percent][data_f_name] = plot_data

            else:
                taxonomy_time_steps[dt_percent] = {data_f_name: plot_data}

        plot_data[data_f_name] = taxonomy_time_steps

# pca_clus_type_list = [['data_type', 'aggl'], ['aggl'], ['data_type'],
        #   ['f_name'],
pca_clus_type_list = [['data_type', 'struc_type']]
#   , ['struc_type', 'aggl']]
for pct in pca_clus_type_list:
    print(pct)
    taxonomy_plotting_pca.plot_taxonomy_pca_over_time(
        taxonomy_time_steps, clus_name_pair=pct)
