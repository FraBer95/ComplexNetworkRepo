import os
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
import seaborn as sns
import matplotlib.pyplot as plt


def statistical_test(data):

    for col in data.columns:
        data[col]=data[col].astype(float)

    corr_mat = data.corr()

    plt.subplots(figsize=(24, 16))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=True, yticklabels=True)
    plt.title("Correlation Matrix")
    plt.savefig("./images/correlation_matrix.png")

    correlation_threshold = 0.60
    high_correlation_pairs = []
    for i in range(len(corr_mat.columns)):
        for j in range(i + 1, len(corr_mat.columns)):
            if abs(corr_mat.iloc[i, j]) > correlation_threshold:
                v1 = corr_mat.columns[i]
                v2 = corr_mat.columns[j]
                correlation_value = corr_mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    drop_list = []
    for pair in high_correlation_pairs:
        print(f"Coppia: {pair[0]} - {pair[1]}, Correlazione: {pair[2]}")
        count_first = len(data[pair[0]].unique())
        count_second = len(data[pair[1]].unique())
        if count_first >= count_second:
                drop_list.append(pair[0])
                print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[0]}")
        else:
                drop_list.append(pair[1])
                print(f"Della Coppia: {pair[0]} - {pair[1]}, con Correlazione: {pair[2]} elimino la Feature {pair[1]}")



    data.drop(columns=drop_list, axis=1, inplace=True)

    threshold = 10
    continuous_features = []
    categorical_features = []

    for column in data.columns:
        unique_values = data[column].nunique()

        if unique_values <= threshold:
            categorical_features.append(column)
        else:
            continuous_features.append(column)


    #computing statistical test
    data_con = data[continuous_features]
    data_cat = data[categorical_features]
    labels = data_cat['Status']

    pvalue_cat = { "Features": [],
                    "P-Value": []
                }

    pvalue_con = { "Features": [],
                    "P-Value": []
                }

    # Chi squared Test

    for feature in data_cat.columns:

        if feature != 'Status':
            contingency = pd.crosstab(data_cat[feature], labels)
            chi2, p, _, _ = chi2_contingency(contingency)
            pvalue_cat["Features"].append(feature)
            pvalue_cat["P-Value"].append(p)

            print(f"Test for {feature} with respect to Target:")
            print(f"Chi2 Value: {chi2}")
            print(f"P-value: {p}")
            print("------------------")

    # Mann-Whitney U Test
    group1 = data_con[labels == 0]
    group2 = data_con[labels == 1]

    for feature in data_con.columns:

        stat, p = mannwhitneyu(group1[feature], group2[feature])
        pvalue_con["Features"].append(feature)
        pvalue_con["P-Value"].append(p)

        print(f"Test for {feature} with respect to Target:")
        print(f"Statistic Value: {stat}")
        print(f"P-value: {p}")
        print("----------------")

    pvalue_cat = pd.DataFrame(pvalue_cat)
    pvalue_con = pd.DataFrame(pvalue_con)

    corrected_pvalue_cat = multipletests(pvalue_cat['P-Value'], method='fdr_bh')[1]
    pvalue_cat['Corrected P-Value'] = corrected_pvalue_cat[:len(pvalue_cat)]
    significant_cat = pvalue_cat[pvalue_cat['Corrected P-Value'] < 0.05]
    not_sign_cat = pvalue_cat[pvalue_cat['Corrected P-Value'] >= 0.05]

    p_val_cat = pd.concat([significant_cat, not_sign_cat], axis=0)
    # p_val_cat.to_csv(r'/Volumes/ExtremeSSD/pycharmProg/surv_OSA/files/categ_pval.csv', index=False)

    if not significant_cat.empty:
        print("Features Categoriche Significative:")
        print(significant_cat)
        print("\n" + "-"*50 + "\n")
    if not not_sign_cat.empty:
        print("Features Categoriche Non Significative:")
        print(not_sign_cat)
        print("\n" + "-" * 50 + "\n")

    corrected_pvalue_con = multipletests(pvalue_con['P-Value'], method='fdr_bh')[1]
    pvalue_con['Corrected P-Value'] = corrected_pvalue_con[:len(pvalue_con)]
    significant_con = pvalue_con[pvalue_con['Corrected P-Value'] < 0.05]
    not_sign_con = pvalue_con[pvalue_con['Corrected P-Value'] >= 0.05]

    p_val_con = pd.concat([significant_con, not_sign_con], axis=0)
    # p_val_con.to_csv(r'/files/con_pval.csv', index=False)

    if not significant_con.empty:
        print("Features Continue Significative:")
        print(significant_con)
        print("\n" + "-"*50 + "\n")
    if not not_sign_con.empty:
        print("Features Continue Non Significative:")
        print(not_sign_con)
        print("\n" + "-" * 50 + "\n")


    # Show continuos features distribution

    sns.set_style('whitegrid')
    for feature in data_con.columns:
        if feature != 'Durata follow-up da dimissione':
            sns.displot(data_con[feature], kde=False, bins = 60)
            folder = os.path.join(r"images/feature_distribution", feature)
            plt.savefig(folder)

    return data





