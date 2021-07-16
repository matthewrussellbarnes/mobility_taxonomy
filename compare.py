import matplotlib.pyplot as plt
import os
import pandas as pd

import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init_lib()

dt = 5000
for _, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'do_not_import']
    for file in files:
        if file.startswith('.'):
            continue

        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)

        plot_name = 'mobility'

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
                network_data, max(network_data.index), dt, data_f_name)

        plot_name = f"{data_f_name}_ti{t}_dti{dt}"
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        taxonomy_plotting.plot_mobility(
            ax, taxonomy_df, plot_name)
        plt.savefig(f"./figs/mobility_{plot_name}.png")

