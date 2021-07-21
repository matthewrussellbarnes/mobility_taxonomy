import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


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


def plot_taxonomy_for_single_network(ax, taxonomy_data, plot_label=''):
    default_plot_params(ax)

    individual = taxonomy_data['individual']
    delta_individual = taxonomy_data['delta_individual']
    neighbourhood = taxonomy_data['neighbourhood']
    delta_neighbourhood = taxonomy_data['delta_neighbourhood']

    mobility, _ = stats.pearsonr(individual, delta_individual)
    assortativity, _ = stats.pearsonr(individual, neighbourhood)
    philanthropy, _ = stats.pearsonr(individual, delta_neighbourhood)
    individuality_community, _ = stats.pearsonr(
        delta_individual, neighbourhood)
    delta_assortativity, _ = stats.pearsonr(
        delta_individual, delta_neighbourhood)
    neighbourhood_mobility, _ = stats.pearsonr(
        neighbourhood, delta_neighbourhood)

    ax.bar(['mobility', 'assortativity', 'philanthropy', 'commmunity', 'delta_assortativity', 'nbrhd_mobility'], [
           mobility, assortativity, philanthropy, individuality_community, delta_assortativity, neighbourhood_mobility])

    ax.set_xlabel('Taxonomy', fontsize=15)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=15)
    ax.set_title(f"Taxonomy for {plot_label}")
    ax.set_ylim([-1.1, 1.1])


def plot_taxonomy_for_multiple_networks(ax, plot_data_dict, dt_percent):
    default_plot_params(ax)

    x_axis_labels = ['mobility', 'assortativity', 'philanthropy',
                     'commmunity', 'delta_assortativity', 'nbrhd_mobility']
    r = np.arange(len(x_axis_labels))
    width = 0.1

    td_i = 0
    even = True
    for data_f_name, plot_data in plot_data_dict.items():

        taxonomy_data = plot_data['taxonomy_data']
        t = plot_data['t']
        dt = plot_data['dt']

        individual = taxonomy_data['individual']
        delta_individual = taxonomy_data['delta_individual']
        neighbourhood = taxonomy_data['neighbourhood']
        delta_neighbourhood = taxonomy_data['delta_neighbourhood']

        mobility, _ = stats.pearsonr(individual, delta_individual)
        assortativity, _ = stats.pearsonr(individual, neighbourhood)
        philanthropy, _ = stats.pearsonr(individual, delta_neighbourhood)
        individuality_community, _ = stats.pearsonr(
            delta_individual, neighbourhood)
        delta_assortativity, _ = stats.pearsonr(
            delta_individual, delta_neighbourhood)
        neighbourhood_mobility, _ = stats.pearsonr(
            neighbourhood, delta_neighbourhood)

        curve_label = f"{data_f_name}: t={t} dt={dt}"

        ax.bar(r + (width * td_i), [
            mobility, assortativity, philanthropy, individuality_community, delta_assortativity, neighbourhood_mobility],
            width=width, label=curve_label)

        if even:
            td_i = abs(td_i)
            td_i += 1
            even = False
        else:
            td_i *= -1
            even = True

    ax.set_xlabel('Taxonomy', fontsize=15)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=15)
    ax.set_title(f"Taxonomy Comparison")
    ax.set_ylim([-1.1, 1.1])
    ax.legend()

    plt.xticks(r + width / len(plot_data_dict), x_axis_labels)
    plt.savefig(f"./figs/taxomony_comparison_all_dtp{dt_percent}.png")
