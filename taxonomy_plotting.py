import matplotlib.pyplot as plt
import datetime

import utilities


def default_plot_params(ax):
    ax.minorticks_on()
    ax.grid(which="major", alpha=1)
    ax.grid(which="minor", alpha=0.2)
    ax.tick_params(axis='both', labelsize=15)


def create_plot_data_file(f_name, x_key, x, y_key, y):
    fig_data = utilities.get_file_data(
        f_name, utilities.figs_data_path)

    if not fig_data:
        fig_data = {x_key: x, y_key: y}

        path = f"{utilities.figs_data_path}/{f_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))}.txt"
        f = open(path, 'w')
        f.write(str(fig_data))
        f.close()


def plot_mobility(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    individual = list(taxonomy_data['individual'])
    delta_individual = list(taxonomy_data['delta_individual'])
    create_plot_data_file(
        f"mobility_{curve_label}", 'individual', individual, 'delta_individual', delta_individual)

    ax.scatter(individual, delta_individual, label=curve_label)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Change in Individual Degree")
    ax.set_title('Individual degree vs change in degree over last 1000i')
    ax.legend()

    plt.savefig(f"./figs/individual_degree_vs_change_{curve_label}.png")
