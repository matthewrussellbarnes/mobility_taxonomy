import os
import re
import pandas as pd

figs_path = os.path.join(os.getcwd(), 'figs/')
taxonomy_data_path = os.path.join(os.getcwd(), 'taxonomies')
dataset_path = os.path.join(os.getcwd(), 'datasets')


def init_lib():
    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    if not os.path.exists(taxonomy_data_path):
        os.mkdir(taxonomy_data_path)


def get_file_path(f_name, path):
    rx = re.compile(f"{f_name}.+")
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != 'archive']
        for file in files:
            if re.match(rx, file):
                f_path = os.path.join(path, file)
                return f_path

    return None
