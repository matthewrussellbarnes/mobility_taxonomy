import matplotlib.pyplot as plt
import os
import csv
import math
import datetime


import ingest_data
import build_taxonomy
import taxonomy_plotting
import utilities

utilities.init()

for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(
        lambda file: not file.startswith('.D'), files)
    for file in filtered_files:
        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)
        data_type = os.path.basename(dirpath)

        data_path = os.path.join(
            utilities.dataset_path, data_type, f"{data_f_name}.csv")
        with open(data_path, encoding='utf-8-sig') as csvfile:
            for max_i_file, _ in enumerate(csvfile):
                pass
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.minorticks_on()
        ax.grid(which="major", alpha=1)
        ax.grid(which="minor", alpha=0.2)
        ax.tick_params(axis='both', labelsize=15)

        with open(data_path, encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ')
            i = 0
            for ri, row in enumerate(reader):
                if ri % math.ceil(max_i_file / 100) == 0:
                    t = row["creation_time"]

                    if '-' in str(t):
                        if 'us_air_traffic' in data_f_name:
                            unix_t = taxonomy_plotting.us_air_date_to_unix(t)
                        else:
                            date_format = "%Y-%m-%d"

                            unix_t = datetime.datetime.timestamp(
                                datetime.datetime.strptime(t, date_format))

                        ax.scatter(i, float(unix_t))
                    else:
                        ax.scatter(i, float(t))

                    i += 1
        plt.savefig(
            f"./figs/dataset_time/{data_f_name.lstrip('.')}.png")
