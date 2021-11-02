import numpy as np


def gini_coeff(imp_dict):
    N = len(imp_dict)
    abs_dif_pairs = 0
    mean_imp = 0
    for _, v1 in imp_dict.items():
        mean_imp += v1
        for _, v2 in imp_dict.items():
            abs_dif_pairs += abs(v1 - v2)
    mean_imp = mean_imp / N

    return abs_dif_pairs / (2 * N * N * mean_imp)


def gini_coefficient(imp_dict):
    x = np.array(list(imp_dict.values()))
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
