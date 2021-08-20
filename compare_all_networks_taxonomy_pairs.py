import matplotlib.pyplot as plt
import os
import pandas as pd
import math

import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init()

dt_percent_list = [10, 25, 50, 75]
for dt_percent in dt_percent_list:
    plot_data = {}
    for _, dirs, files in os.walk(utilities.dataset_path, topdown=True):
        dirs[:] = [d for d in dirs if d != 'archive']
        filtered_files = filter(lambda file: not file.startswith('.'), files)
        for file in filtered_files:
            data_f_name = os.path.splitext(file)[0]
            print(data_f_name)

            taxonomy_data_f_path = utilities.get_file_path(
                f"{data_f_name}_dtp{dt_percent}", utilities.taxonomy_data_path)

            if taxonomy_data_f_path:
                t = taxonomy_data_f_path[
                    taxonomy_data_f_path.index("_ti") + 3:
                    taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_ti") + 3)]
                taxonomy_df = pd.read_csv(taxonomy_data_f_path)

            else:
                network_data = ingest_data.ingest_data(data_f_name)
                taxonomy_df, t = build_taxonomy.build_taxonomy(
                    network_data, max(network_data.index), dt_percent, data_f_name)

            dt = int(math.ceil(int(t) * (dt_percent / 100)))

            plot_data[data_f_name] = {
                'taxonomy_data': taxonomy_df,
                't': t,
                'dt': dt
            }

    taxonomy_plotting.plot_taxonomy_pairs_for_multiple_networks(
        plot_data, dt_percent)
