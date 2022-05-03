import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import numpy as np

import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init()

dt_percent_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for dt_percent in dt_percent_list:
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    plot_data = {}
    for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
        dirs[:] = [d for d in dirs if d != 'archive']
        filtered_files = filter(lambda file: not file.startswith('.'), files)
        for file in filtered_files:
            data_f_name = os.path.splitext(file)[0]
            print(data_f_name)
            data_type = os.path.basename(dirpath)

            taxonomy_data_f_path = utilities.get_file_path(
                f"{data_f_name}_dtp{dt_percent}", utilities.taxonomy_data_path)

            if taxonomy_data_f_path:
                t = taxonomy_data_f_path[
                    taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2:
                    taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2)]
                taxonomy_df = pd.read_csv(taxonomy_data_f_path)

            else:
                network_data = ingest_data.ingest_data(data_f_name)
                taxonomy_df, t = build_taxonomy.build_taxonomy(
                    network_data, max(network_data.index), dt_percent, data_f_name)

            dt = int(math.ceil(int(t) * (dt_percent / 100)))

            plot_data[data_f_name] = {
                'taxonomy_data': taxonomy_df,
                't': t,
                'dt': dt,
                'data_type': data_type,
                'struc_type': 'struc_type'
            }

    taxonomy_plotting.plot_taxonomy_for_multiple_networks(
        ax, plot_data, dt_percent)
