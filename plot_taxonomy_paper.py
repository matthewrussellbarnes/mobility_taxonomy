import os
import math

import MobilityTaxonomy
import taxonomy_plotting
import taxonomy_plotting_pca
import utilities

utilities.init()

dt_percent_list = [utilities.dt_percent_list[0]]
mt_list = []
for dirpath, dirs, files in os.walk(utilities.dataset_path, topdown=True):
    dirs[:] = [d for d in dirs if d != 'archive']
    filtered_files = filter(lambda file: not file.startswith('.'), files)
    for file in filtered_files:
        mt = MobilityTaxonomy.MobilityTaxonomy(
            file,
            dt_percent_list,
            os.path.basename(dirpath),
            utilities.structure_type_lookup[os.path.splitext(file)[0]])
        mt.build(save=True)
        mt_list.append(mt)

for dt_percent in dt_percent_list:
    plot_data = {}
    for mob_tax in mt_list:
        plot_data[mob_tax.networkf] = {
            'taxonomy_data': mob_tax.taxonomy_df_dict[dt_percent],
            'stats_data': mob_tax.stats_df,
            't': mob_tax.t,
            'dt': int(math.ceil(int(mob_tax.t) * (dt_percent / 100))),
            'dt_percent': dt_percent,
            'data_type': mob_tax.data_type,
            'struc_type': mob_tax.struc_type
        }

    pca_clus_type_list = [['aggl']]
    # [['data_type', 'aggl'], ['aggl'], ['data_type'],
    #  ['f_name'], ['data_type', 'struc_type'], ['struc_type', 'aggl']]
    for pct in pca_clus_type_list:
        print(pct)
        taxonomy_plotting_pca.plot_taxonomy_pca(
            plot_data, dt_percent, clus_name_pair=pct)

    equality_type_list = ['norm_it', 'norm_time']
    equality_clus_type_list = ['data_type', 'struc_type']
    for eqt in equality_type_list:
        for clt in equality_clus_type_list:
            print(eqt, clt)
            taxonomy_plotting.plot_equality(
                plot_data, y_type=eqt, clustering_type=clt)
