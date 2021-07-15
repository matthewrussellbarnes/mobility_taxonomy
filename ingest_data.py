import os
import csv
import pandas as pd


def ingest_data(data_f_name, first_row, max_rows):
    data_path = os.path.join(os.getcwd(), f"datasets/{data_f_name}.csv")

    network_df = pd.DataFrame(columns=['n1', 'n2', 'creation_time'])
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        r = 0
        for row in reader:
            if r >= first_row:
                if r < first_row + max_rows:
                    n1 = row["n1"]
                    n2 = row["n2"]
                    creation_time = row["creation_time"]

                    network_df.loc[len(network_df.index)] = [
                        n1, n2, creation_time]
                else:
                    break

    return network_df
