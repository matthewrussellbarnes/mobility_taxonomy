import os
import pandas as pd

import ingest
import utilities


class MobilityTaxonomy:
    def __init__(self, network_file_name, dt_percent_list, data_type, struc_type, max_rows=1000000):
        self.networkf = network_file_name
        self.data_type = data_type
        self.struc_type = struc_type
        self.dt_percent_list = dt_percent_list
        self.max_rows = max_rows

    def build(self):
        taxonomy_df_dict = {}
        data_f_name = os.path.splitext(self.networkf)[0]
        print(data_f_name)

        build_taxonomy_dt_list = []
        for dt_percent in self.dt_percent_list:
            taxonomy_data_f_path = utilities.get_file_path(
                f"{data_f_name}_dtp{dt_percent}", utilities.taxonomy_data_path)

            if taxonomy_data_f_path:
                taxonomy_df_dict[dt_percent] = pd.read_csv(
                    taxonomy_data_f_path)
                self.t = taxonomy_data_f_path[
                    taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2:
                    taxonomy_data_f_path.index("_", taxonomy_data_f_path.index("_e", taxonomy_data_f_path.index("_dtp")) + 2)]
            else:
                build_taxonomy_dt_list.append(dt_percent)

        if build_taxonomy_dt_list:
            build_taxonomy_df_dict, self.stats_df, self.t = ingest.build_taxonomy(
                self.networkf, build_taxonomy_dt_list, max_rows=self.max_rows)
            taxonomy_df_dict = taxonomy_df_dict | build_taxonomy_df_dict
        else:
            stats_data_f_path = utilities.get_file_path(
                f"{data_f_name}", utilities.stats_data_path)
            self.stats_df = pd.read_csv(stats_data_f_path)

        self.taxonomy_df_dict = taxonomy_df_dict
