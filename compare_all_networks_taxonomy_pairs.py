import matplotlib.pyplot as plt
import os
import pandas as pd
import math

import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init()

dt_percent_list = [10]  # , 25, 50, 75]
for dt_percent in dt_percent_list:
    plot_data = {}
    for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
        dirs[:] = [d for d in dirs if d != 'archive']
        filtered_files = filter(lambda file: not file.startswith('.'), files)
        for file in filtered_files:
            data_f_name = os.path.splitext(file)[0]
            print(data_f_name)
            data_type = os.path.basename(dirpath)
            struc_type = utilities.structure_type_lookup[data_f_name]

            taxonomy_data_f_path = utilities.get_file_path(
                f"{data_f_name}_dtp{dt_percent}", utilities.taxonomy_data_path)
            stats_data_f_path = utilities.get_file_path(
                f"{data_f_name}", utilities.stats_data_path)

            t = taxonomy_data_f_path[
                taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2:
                taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2)]
            taxonomy_df = pd.read_csv(taxonomy_data_f_path)

            stats_df = pd.read_csv(stats_data_f_path)

            plot_data[data_f_name] = {
                'taxonomy_data': taxonomy_df,
                'stats_data': stats_df,
                't': t,
                'dt_percent': dt_percent,
                'dt': int(math.ceil(int(t) * (dt_percent / 100))),
                'data_type': data_type,
                'struc_type': struc_type
            }

    taxonomy_plotting.plot_taxonomy_pairs_for_multiple_networks(
        plot_data, dt_percent)
