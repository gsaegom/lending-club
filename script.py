import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, chi2
from sklearn.feature_selection import mutual_info_classif, \
    mutual_info_regression
from statsmodels.stats.weightstats import ztest

sub_grade_mapping = {
    'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
    'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
    'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
    'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
    'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
    'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
    'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35
}


def perform_chi2_test(df, column, alpha=0.05):
    contingency_table = pd.crosstab(df['Status'], df[column])
    chi2_stat, p, _, _ = chi2_contingency(contingency_table)

    degrees_of_freedom = (contingency_table.shape[0] - 1) * (
            contingency_table.shape[1] - 1)

    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)

    lower_bound = max(0,
                      p - critical_value * (p * (1 - p) / df.shape[0]) ** 0.5)
    upper_bound = min(1,
                      p + critical_value * (p * (1 - p) / df.shape[0]) ** 0.5)

    print(f"{column}: Chi-square value: {chi2_stat}, p-value: {p}, "
          f"confidence interval: ({lower_bound}, {upper_bound})")


def perform_ztest(df, column, alternative='smaller'):
    group1 = df.dropna(subset=[column])[df['Status'] == 0][column]
    group2 = df.dropna(subset=[column])[df['Status'] == 1][column]
    statistic, p = ztest(group1, group2, alternative=alternative)
    confidence_interval = confidence_interval_two_means(group1, group2, 95,
                                                        equal_var=True)
    print(f'{column}: z-statistic: {statistic}, p-value: {p}, confidence '
          f'interval 95%: {confidence_interval}')


def perform_anova(df, column, alpha=0.05, variable='Status'):
    unique_values = df[column].unique()
    groups = [df[df[column] == value][variable] for value in unique_values]
    f_statistic, p_value = f_oneway(*groups)
    df_between = len(groups) - 1
    df_within = df.shape[0] - len(groups)

    # Calculate the critical value for the given alpha level
    critical_value = stats.f.ppf(1 - alpha, df_between, df_within)

    # Calculate the confidence interval for the p-value
    lower_bound = max(0, p_value - critical_value * (
            p_value * (1 - p_value) / df.shape[0]) ** 0.5)
    upper_bound = min(1, p_value + critical_value * (
            p_value * (1 - p_value) / df.shape[0]) ** 0.5)

    print(f"{column}: F-statistic: {f_statistic}, p-value: {p_value}, "
          f"confidence interval: ({lower_bound}, {upper_bound})")


def make_mi_scores(X, y, discrete_features='auto', model='classification'):
    X.ffill(inplace=True)
    if model == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42,
                                        discrete_features=discrete_features)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42,
                                           discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def critical_value(alpha: int, dist='z', case='two-tailed', samp_sz=None):
    if dist == 'z' and case == 'two-tailed':
        crit_val = stats.norm.ppf(1 - alpha / 2)
    elif dist == 'z' and case == 'one-tailed':
        crit_val = stats.norm.ppf(1 - alpha)
    elif dist == 't' and case == 'two-tailed':
        crit_val = stats.t.ppf(1 - alpha / 2, samp_sz - 1)
    elif dist == 't' and case == 'one-tailed':
        crit_val = stats.t.ppf(1 - alpha, samp_sz - 1)
    else:
        crit_val = None
    return crit_val


def confidence_interval_two_means(x, y, ci, equal_var=False):
    n1 = len(x)
    n2 = len(y)
    mu1 = np.mean(x)
    mu2 = np.mean(y)
    bst_est = mu1 - mu2
    alpha = 1 - ci * 1e-2
    crit_val = critical_value(alpha)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    if equal_var:
        se = math.sqrt(v1 / n1 + v2 / n2)  # pooled approach
    else:
        se = math.sqrt(
            ((n1 - 1) * v1 + (n2 - 2) * v2) / (n1 + n2 - 2)) * math.sqrt(
            1 / n1 + 1 / n2)  # unpooled approach
    lcb = bst_est - crit_val * se
    ucb = bst_est + crit_val * se
    return lcb, ucb


def filter_columns(df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """
    Filter columns from DataFrame based on the percentage of empty strings and NaN values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): Maximum percentage of empty strings and NaN values allowed to keep the column. Default is 0.25 (25%).

    Returns:
    - pd.DataFrame: DataFrame with filtered columns.
    """

    null_empty_percentage = df.eq('').mean() + df.isnull().mean()

    filtered_columns = null_empty_percentage[
        null_empty_percentage < threshold].index

    return df[filtered_columns]


def downgrade_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns of DataFrame to appropriate data types.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with converted columns.
    """

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')
        df[col] = df[col].astype('Int64', errors='ignore')
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype('category', errors='ignore')

    return df


import pandas as pd


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary columns in DataFrame by replacing categorical values with 1 and 0.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with binary columns encoded.
    """
    for col in df.select_dtypes(include='object'):
        unique_values = df[col].unique()
        if len(unique_values) == 2:
            true_value = unique_values[0]
            false_value = unique_values[1]
            new_col_name = f'{col}_{true_value.replace(" ", "_")}'
            df[new_col_name] = df[col].replace(
                {true_value: 1, false_value: 0})
            df.drop(col, axis=1, inplace=True)
    return df


import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words()
stop_words.append('&')


def get_most_common_words(column, number_word_combination=1,
                          number_results=50) -> pd.Series:
    """
    Extracts the most common words or word combinations from a text column.

    Parameters:
    - column: pd.Series
        A pandas Series containing the text data.
    - number_word_combination: int, optional (default=1)
        The number of words to consider as a combination.
    - number_results: int, optional (default=50)
        The number of most common words to return.

    Returns:
    - pd.Series
        A pandas Series containing the counts of the most common words or 
        word combinations.
    """

    words = " ".join(column.str.lower()).split()
    word_combos = []
    word_combo = []
    for i in range(len(words) - (number_word_combination - 1)):
        for j in range(number_word_combination):
            word_combo.append(words[i + j])
        if set(word_combo).isdisjoint(stop_words):
            word_combos.append(' '.join(word_combo))
        word_combo.clear()
    return pd.Series(word_combos).value_counts()[:number_results]


def one_hot_encode_common_words(df: pd.DataFrame, column_name: str,
                                most_common_words: pd.Series | list) -> \
        pd.DataFrame:
    """
    One-hot encodes the most common words or word combinations from a specified column in a DataFrame and integrates them into the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to be encoded.
        column_name (str): The name of the column to be encoded.
        most_common_words (pd.Series): A Series containing the most common words or word combinations and their frequencies.

    Returns:
        pd.DataFrame: The DataFrame with the specified column one-hot encoded.
    """

    df_employment_title = pd.DataFrame()
    if isinstance(most_common_words, pd.Series):
        most_common_words = most_common_words.index
    for word in most_common_words:
        df_employment_title[column_name + '_' + word.title()] = \
            df[column_name].apply(
                lambda title: 1 if word.lower()
                                   in title.lower() else 0)

    df_employment_title = downgrade_column_types(df_employment_title)
    df = df.merge(df_employment_title, left_index=True, right_index=True)
    df.drop(column_name, axis=1, inplace=True)

    return df


def map_sub_grade(column: pd.Series) -> pd.Series:
    """
    Maps sub-grade values to their corresponding integer representations using a predefined mapping dictionary.

    Parameters:
        column (pd.Series): A pandas Series containing sub-grade values to be mapped.

    Returns:
        pd.Series: A pandas Series with sub-grade values mapped to their corresponding integer representations.
    """

    return column.map(sub_grade_mapping).astype(int)


def reverse_map_sub_grade(column: pd.Series) -> pd.Series:
    """
    Reverses the mapping of integer representations back to their original sub-grade values using the same mapping dictionary.

    Parameters:
        column (pd.Series): A pandas Series containing integer representations of sub-grade values.

    Returns:
        pd.Series: A pandas Series with integer representations of sub-grade values reversed to their original sub-grade values.
    """

    reverse_sub_grade_mapping = {value: key for key, value in
                                 sub_grade_mapping.items()}

    return column.map(reverse_sub_grade_mapping)