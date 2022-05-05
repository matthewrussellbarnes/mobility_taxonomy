import os
import re
import matplotlib.pyplot as plt
from matplotlib import colors
import string
import json

# {"dataset_name": ["collection_type", "structure_type"]}
with open('dataset_type_lookup.json') as f:
    dataset_type_lookup = json.load(f)
    f.close()

figs_path = os.path.join(os.getcwd(), 'figs/')
taxonomy_data_path = os.path.join(os.getcwd(), 'taxonomies')
stats_data_path = os.path.join(os.getcwd(), 'network_stats')
dataset_path = os.path.join(os.getcwd(), 'datasets')

dt_percent_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

cmap20 = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
                                '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000'])
cmap10 = colors.ListedColormap(['#ffe119', '#4363d8', '#f58231',
                                '#dcbeff', '#800000', '#000075', '#a9a9a9', '#9A6324', '#fabed4'])

plot_markers = ["o", "s", "D", "*", 'P', 'X', '1', 'v', 'p', '$£$']
plot_letters = list(string.ascii_letters)

plot_font_size = 22


def init():
    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    if not os.path.exists(taxonomy_data_path):
        os.mkdir(taxonomy_data_path)
    if not os.path.exists(stats_data_path):
        os.mkdir(stats_data_path)
    plt.rcParams.update(
        {'axes.labelsize': plot_font_size, 'axes.titlesize': plot_font_size, 'font.size': 18})


def get_file_path(f_name, path):
    rx = re.compile(f"{f_name}.+")
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != 'archive']
        for file in files:
            if re.match(rx, file):
                f_path = os.path.join(path, file)
                return f_path

    return None


def get_file_path_for_multiple(f_name, path):
    f_path_list = []
    rx = re.compile(f"{f_name}.+")
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != 'archive']
        for file in files:
            if re.match(rx, file):
                f_path = os.path.join(path, file)
                f_path_list.append(f_path)

    return f_path_list


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
