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

dt_percent = 50
for _, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'do_not_import']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)

        taxonomy_data_f_path = utilities.get_file_path(
            data_f_name, utilities.taxonomy_data_path)

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
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        taxonomy_plotting.plot_taxonomy_for_single_network(
            ax, taxonomy_df, f"{data_f_name} with t={t} and dt={dt}")
        plt.savefig(
            f"./figs/taxomony_{data_f_name}_ti{t}_dti{dt}.png")
