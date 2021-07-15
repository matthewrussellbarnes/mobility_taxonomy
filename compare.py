import matplotlib.pyplot as plt
import os

import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init_lib()

for _, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'do_not_import']
    for file in files:
        if file.startswith('.'):
            continue

        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)

        plot_name = 'mobility'

        f_name = f"{plot_name}_{data_f_name}"
        fig_data = utilities.get_file_data(
            f_name, utilities.figs_data_path)

        if not fig_data:
            network_data = ingest_data.ingest_data(data_f_name)
            fig_data = build_taxonomy.build_taxonomy(
                network_data, max(network_data.index), 1000)

        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        taxonomy_plotting.plot_mobility(ax, fig_data, data_f_name)
