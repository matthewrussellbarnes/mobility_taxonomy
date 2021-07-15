import matplotlib.pyplot as plt
import os
import re

from numpy.core.fromnumeric import trace

import ingest_data
import build_taxonomy
import taxonomy_plotting

f_name = 'CollegeMsg'
network_data = ingest_data.ingest_data(f_name)
taxonomy_data = build_taxonomy.build_taxonomy(
    network_data, max(network_data.index), 1000)


_, ax = plt.subplots(1, 1, figsize=(15, 10))
taxonomy_plotting.plot_mobility(ax, taxonomy_data, f_name)
