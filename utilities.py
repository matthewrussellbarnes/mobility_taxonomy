import os
import re

figs_path = os.path.join(os.getcwd(), 'figs/')
figs_data_path = os.path.join(os.getcwd(), 'figs/figs_data')
dataset_path = os.path.join(os.getcwd(), 'datasets')


def init_lib():
    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    if not os.path.exists(figs_data_path):
        os.mkdir(figs_data_path)


def get_file_data(f_name, path):
    rx = re.compile(f"{f_name}.+")
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != 'archive']
        for file in files:
            if re.match(rx, file):
                f_path = os.path.join(path, file)
                f = open(f_path, 'r')
                f_data = eval(f.read())
                f.close()
                return f_data

    return None
