def gini_coeff(importance):
    gini_coeff = {}
    for i, imp_dict in importance.items():
        N = len(imp_dict)
        abs_dif_pairs = 0
        mean_imp = 0
        for _, v1 in imp_dict.items():
            mean_imp += v1
            for _, v2 in imp_dict.items():
                abs_dif_pairs += abs(v1 - v2)
        mean_imp = mean_imp / N
        gini_coeff[i] = abs_dif_pairs / (2 * N * N * mean_imp)

    return gini_coeff
