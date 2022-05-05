import os

import ingest
import utilities

utilities.init()

max_rows = 1000000
for _, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        data_f_name = os.path.splitext(file)[0]
        print(data_f_name)
        mt = ingest.MobilityTaxonomy(
            file,
            utilities.dt_percent_list,
            utilities.dataset_type_lookup[os.path.splitext(file)[0]][0],
            utilities.dataset_type_lookup[os.path.splitext(file)[0]][1],
            max_rows=max_rows)
        mt.build(save=True)
