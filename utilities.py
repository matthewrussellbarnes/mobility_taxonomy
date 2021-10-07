import os
import re
import matplotlib.pyplot as plt

figs_path = os.path.join(os.getcwd(), 'figs/')
taxonomy_data_path = os.path.join(os.getcwd(), 'taxonomies')
stats_data_path = os.path.join(os.getcwd(), 'network_stats')
dataset_path = os.path.join(os.getcwd(), 'datasets')


def init():
    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    if not os.path.exists(taxonomy_data_path):
        os.mkdir(taxonomy_data_path)
    if not os.path.exists(stats_data_path):
        os.mkdir(stats_data_path)
    plt.rcParams.update(
        {'axes.labelsize': 'large', 'axes.titlesize': 'xx-large'})


def get_file_path(f_name, path):
    rx = re.compile(f"{f_name}.+")
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != 'archive']
        for file in files:
            if re.match(rx, file):
                f_path = os.path.join(path, file)
                return f_path

    return None
