import os
import csv
import pandas as pd

import utilities


def ingest_data(data_f_name, first_row=None, max_rows=None):
    data_path = os.path.join(utilities.dataset_path, f"{data_f_name}.csv")

    network_df = pd.DataFrame(columns=['n1', 'n2', 'creation_time'])
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        r = 0
        for row in reader:
            if not first_row or r >= first_row:
                if not first_row:
                    max_row = max_rows
                elif max_rows:
                    max_row = first_row + max_rows
                if not max_row or r < max_row:
                    n1 = row["n1"]
                    n2 = row["n2"]
                    creation_time = row["creation_time"]

                    network_df.loc[len(network_df.index)] = [
                        n1, n2, creation_time]
                else:
                    break
            r += 1
    return network_df
