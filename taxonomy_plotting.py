import matplotlib.pyplot as plt
import datetime

import utilities


def default_plot_params(ax):
    ax.minorticks_on()
    ax.grid(which="major", alpha=1)
    ax.grid(which="minor", alpha=0.2)
    ax.tick_params(axis='both', labelsize=15)


def plot_mobility(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    individual = taxonomy_data['individual']
    delta_individual = taxonomy_data['delta_individual']

    ax.scatter(individual, delta_individual, label=curve_label)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Change in Individual Degree")
    ax.set_title('Individual Mobility')
    ax.legend()


def plot_neighbourhood_mobility(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    neighbourhood = taxonomy_data['neighbourhood']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    ax.scatter(neighbourhood, delta_neighbourhood, label=curve_label)
    ax.set_xlabel("Neighbourhood Degree")
    ax.set_ylabel("Change in Neighbourhood Degree")
    ax.set_title('Neighbourhood Mobility')
    ax.legend()


def plot_assortativity(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    individual = taxonomy_data['individual']
    neighbourhood = taxonomy_data['neighbourhood']

    ax.scatter(individual, neighbourhood, label=curve_label)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Average Neighbourhood Degree")
    ax.set_title('Assortativity')
    ax.legend()


def plot_delta_assortativity(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    delta_individual = taxonomy_data['delta_individual']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    ax.scatter(delta_individual, delta_neighbourhood, label=curve_label)
    ax.set_xlabel("Change in Individual Degree")
    ax.set_ylabel("Change in Average Neighbourhood Degree")
    ax.set_title('Change in Assortativity')
    ax.legend()


def plot_philanthropy(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    individual = taxonomy_data['individual']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    ax.scatter(individual, delta_neighbourhood, label=curve_label)
    ax.set_xlabel("Individual Degree")
    ax.set_ylabel("Change in Average Neighbourhood Degree")
    ax.set_title('Philanthropy')
    ax.legend()


def plot_individuality_vs_community(ax, taxonomy_data, curve_label=''):
    default_plot_params(ax)

    delta_individual = taxonomy_data['delta_individual']
    neighbourhood = taxonomy_data['neighbourhood']

    ax.scatter(delta_individual, neighbourhood, label=curve_label)
    ax.set_xlabel("Change in Individual Degree")
    ax.set_ylabel("Average Neighbourhood Degree")
    ax.set_title('Individuality/Community')
    ax.legend()

    plt.savefig(f"./figs/individual_degree_vs_change_{curve_label}.png")
