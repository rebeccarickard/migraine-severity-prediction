"""Reusable functions for exploratory data analysis."""

import matplotlib.pyplot as plt
import pandas as pd

from collections.abc import Sequence
from scipy.stats import chi2_contingency


def validate_binary_columns(
    data: pd.DataFrame,
    variables: Sequence[str],
) -> dict[str, list]:
    """
    Check that intended binary columns contain only 0 and 1.

    Returns
    -------
    dict
        Variables with missing columns or unexpected values.
    """

    problems = {}

    for variable in variables:
        if variable not in data.columns:
            problems[variable] = ["COLUMN NOT FOUND"]
            continue

        observed_values = set(
            data[variable].dropna().unique().tolist()
        )

        unexpected_values = sorted(
            observed_values - {0, 1, 0.0, 1.0}
        )

        if unexpected_values:
            problems[variable] = unexpected_values

    return problems


def create_binary_prevalence_table(
    data: pd.DataFrame,
    variables: Sequence[str],
    target: str,
    severity_order: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate the percentage of observations in which each binary
    factor equals 1 within each severity category.
    """

    output_rows = []

    for variable in variables:
        percentages = (
            data.groupby(
                target,
                observed=False,
            )[variable]
            .mean()
            .mul(100)
            .reindex(severity_order)
        )

        for severity, percentage in percentages.items():
            output_rows.append({
                "Feature": variable,
                "Severity": severity,
                "Percent Present": round(percentage, 2),
            })

    return pd.DataFrame(output_rows)


def plot_binary_prevalence(
    data: pd.DataFrame,
    variables: Sequence[str],
    target: str,
    severity_order: Sequence[str],
    group_name: str,
) -> None:
    """
    Create one bar plot per binary feature.

    Each plot shows the percentage of migraine episodes in which
    the factor was present within each severity category.
    """

    for variable in variables:
        percentages = (
            data.groupby(
                target,
                observed=False,
            )[variable]
            .mean()
            .mul(100)
            .reindex(severity_order)
        )

        figure, axis = plt.subplots(figsize=(7, 4))

        percentages.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_title(f"{group_name}: {variable}")
        axis.set_xlabel("Migraine severity")
        axis.set_ylabel("Episodes with factor present (%)")
        axis.set_ylim(0, 100)
        axis.tick_params(axis="x", rotation=0)

        figure.tight_layout()
        plt.show()


def create_ordinal_percentage_table(
    data: pd.DataFrame,
    variable: str,
    target: str,
    severity_order: Sequence[str],
) -> pd.DataFrame:
    """
    Create row-normalized percentages for an ordinal predictor.
    """

    table = pd.crosstab(
        data[target],
        data[variable],
        normalize="index",
    ).mul(100)

    return table.reindex(severity_order)


def plot_grouped_binary_prevalence(
    data: pd.DataFrame,
    variables: Sequence[str],
    target: str,
    severity_order: Sequence[str],
    group_name: str,
) -> pd.DataFrame:
    """
    Plot all binary variables in one grouped horizontal bar chart.

    Each bar represents the percentage of migraine episodes in which
    the factor was present within a severity category.

    Returns
    -------
    pd.DataFrame
        Wide-format prevalence table used to create the plot.
    """

    prevalence_rows = []

    for variable in variables:
        percentages = (
            data.groupby(
                target,
                observed=False,
            )[variable]
            .mean()
            .mul(100)
            .reindex(severity_order)
        )

        row = {
            "Feature": variable,
            **percentages.to_dict(),
        }

        prevalence_rows.append(row)

    prevalence_wide = (
        pd.DataFrame(prevalence_rows)
        .set_index("Feature")
    )

    figure_height = max(5, len(variables) * 0.5)

    axis = prevalence_wide.plot(
        kind="barh",
        figsize=(10, figure_height),
    )

    axis.set_title(
        f"Prevalence of {group_name} by Migraine Severity"
    )
    axis.set_xlabel("Episodes with factor present (%)")
    axis.set_ylabel("")
    axis.set_xlim(0, 100)
    axis.legend(
        title="Migraine severity",
        labels=severity_order,
    )

    plt.tight_layout()
    plt.show()

    return prevalence_wide


"""Reusable functions for exploratory data analysis."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def analyze_binary_associations(
    data: pd.DataFrame,
    variables: Sequence[str],
    target: str,
    severity_order: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Test associations between binary predictors and a categorical target.

    For each binary predictor, this function:
    1. Creates an observed-count contingency table.
    2. Creates a row-normalized percentage table.
    3. Performs a chi-square test of independence.
    4. Calculates Cramér's V.
    5. Checks whether expected cell counts satisfy common assumptions.

    Parameters
    ----------
    data
        DataFrame containing the predictors and target.
    variables
        Names of binary predictor columns.
    target
        Name of the categorical outcome column.
    severity_order
        Desired order of target categories.

    Returns
    -------
    results
        One-row-per-feature summary of statistical results.
    crosstabs
        Dictionary containing count and percentage tables for each feature.
    """

    results_rows = []
    crosstabs = {}

    for variable in variables:
        if variable not in data.columns:
            results_rows.append({
                "Feature": variable,
                "N": np.nan,
                "Chi-Square": np.nan,
                "Degrees of Freedom": np.nan,
                "P-Value": np.nan,
                "Cramers V": np.nan,
                "Minimum Expected Count": np.nan,
                "Expected Counts Below 5": np.nan,
                "Assumptions Met": False,
                "Status": "Column not found",
            })
            continue

        analysis_data = data[[target, variable]].dropna()

        observed = pd.crosstab(
            analysis_data[target],
            analysis_data[variable],
        ).reindex(severity_order, fill_value=0)

        # A chi-square test requires at least two observed categories
        # for both the predictor and the target.
        if observed.shape[0] < 2 or observed.shape[1] < 2:
            results_rows.append({
                "Feature": variable,
                "N": len(analysis_data),
                "Chi-Square": np.nan,
                "Degrees of Freedom": np.nan,
                "P-Value": np.nan,
                "Cramers V": np.nan,
                "Minimum Expected Count": np.nan,
                "Expected Counts Below 5": np.nan,
                "Assumptions Met": False,
                "Status": "Insufficient category variation",
            })

            crosstabs[variable] = {
                "counts": observed,
                "row_percentages": (
                    observed.div(observed.sum(axis=1), axis=0) * 100
                ),
            }
            continue

        chi2, p_value, dof, expected = chi2_contingency(
            observed,
            correction=False,
        )

        sample_size = observed.to_numpy().sum()
        rows, columns = observed.shape

        denominator = min(rows - 1, columns - 1)

        cramers_v = (
            np.sqrt((chi2 / sample_size) / denominator)
            if sample_size > 0 and denominator > 0
            else np.nan
        )

        minimum_expected = expected.min()
        expected_below_five = int((expected < 5).sum())
        expected_below_one = int((expected < 1).sum())
        total_expected_cells = expected.size

        proportion_below_five = (
            expected_below_five / total_expected_cells
        )

        # Common rule:
        # - no expected cell should be below 1
        # - no more than 20% should be below 5
        assumptions_met = (
            expected_below_one == 0
            and proportion_below_five <= 0.20
        )

        percentage_table = (
            observed
            .div(observed.sum(axis=1), axis=0)
            .mul(100)
            .round(2)
        )

        expected_table = pd.DataFrame(
            expected,
            index=observed.index,
            columns=observed.columns,
        )

        crosstabs[variable] = {
            "counts": observed,
            "row_percentages": percentage_table,
            "expected_counts": expected_table,
        }

        results_rows.append({
            "Feature": variable,
            "N": sample_size,
            "Chi-Square": chi2,
            "Degrees of Freedom": dof,
            "P-Value": p_value,
            "Cramers V": cramers_v,
            "Minimum Expected Count": minimum_expected,
            "Expected Counts Below 5": expected_below_five,
            "Assumptions Met": assumptions_met,
            "Status": "Test completed",
        })

    results = pd.DataFrame(results_rows)

    return results, crosstabs


def add_benjamini_hochberg_correction(
    results: pd.DataFrame,
    p_value_column: str = "P-Value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply the Benjamini-Hochberg false-discovery-rate correction.
    """

    corrected = results.copy()

    valid_mask = corrected[p_value_column].notna()
    valid_results = corrected.loc[valid_mask].copy()

    number_of_tests = len(valid_results)

    if number_of_tests == 0:
        corrected["Adjusted P-Value"] = np.nan
        corrected["Significant After FDR"] = False
        return corrected

    valid_results = valid_results.sort_values(
        p_value_column
    ).copy()

    ranks = np.arange(1, number_of_tests + 1)

    raw_adjusted = (
        valid_results[p_value_column].to_numpy()
        * number_of_tests
        / ranks
    )

    # Enforce monotonic adjusted p-values.
    adjusted = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    valid_results["Adjusted P-Value"] = adjusted

    corrected["Adjusted P-Value"] = np.nan

    corrected.loc[
        valid_results.index,
        "Adjusted P-Value",
    ] = valid_results["Adjusted P-Value"]

    corrected["Significant After FDR"] = (
        corrected["Adjusted P-Value"] < alpha
    ).fillna(False)

    return corrected


def analyze_categorical_associations(
    data: pd.DataFrame,
    variable: str,
    target: str,
    target_order: Sequence[str] | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """
    Test the association between a categorical predictor and a categorical target.

    This function:
    1. Creates an observed-count contingency table.
    2. Creates a row-normalized percentage table.
    3. Performs a chi-square test of independence.
    4. Calculates Cramér's V.
    5. Evaluates expected cell counts to assess chi-square assumptions.

    Parameters
    ----------
    data
        DataFrame containing the predictor and target.
    variable
        Name of the categorical predictor column.
    target
        Name of the categorical outcome column.
    target_order
        Optional desired order of target categories.

    Returns
    -------
    results
        Dictionary containing statistical results and assumption checks.
    crosstabs
        Dictionary containing observed counts, row percentages,
        and expected counts.
    """

    analysis_data = data[[target, variable]].dropna()

    observed = pd.crosstab(
        analysis_data[target],
        analysis_data[variable],
    )

    if target_order is not None:
        observed = observed.reindex(target_order, fill_value=0)

    # Chi-square requires at least two categories
    # for both the predictor and target.
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        results = {
            "Feature": variable,
            "N": len(analysis_data),
            "Chi-Square": np.nan,
            "Degrees of Freedom": np.nan,
            "P-Value": np.nan,
            "Cramers V": np.nan,
            "Minimum Expected Count": np.nan,
            "Expected Counts Below 5": np.nan,
            "Expected Counts Below 1": np.nan,
            "Proportion Expected Below 5": np.nan,
            "Assumptions Met": False,
            "Status": "Insufficient category variation",
        }

        crosstabs = {
            "counts": observed,
            "row_percentages": (
                observed
                .div(observed.sum(axis=1), axis=0)
                .mul(100)
                .round(2)
            ),
        }

        return results, crosstabs

    chi2, p_value, dof, expected = chi2_contingency(
        observed,
        correction=False,
    )

    sample_size = observed.to_numpy().sum()
    rows, columns = observed.shape

    denominator = min(rows - 1, columns - 1)

    cramers_v = (
        np.sqrt((chi2 / sample_size) / denominator)
        if sample_size > 0 and denominator > 0
        else np.nan
    )

    expected_table = pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns,
    )

    minimum_expected = expected.min()
    expected_below_five = int((expected < 5).sum())
    expected_below_one = int((expected < 1).sum())
    total_expected_cells = expected.size

    proportion_below_five = (
        expected_below_five / total_expected_cells
    )

    # Common chi-square assumption guideline:
    # - no expected count below 1
    # - no more than 20% of expected counts below 5
    assumptions_met = (
        expected_below_one == 0
        and proportion_below_five <= 0.20
    )

    percentage_table = (
        observed
        .div(observed.sum(axis=1), axis=0)
        .mul(100)
        .round(2)
    )

    results = {
        "Feature": variable,
        "N": sample_size,
        "Chi-Square": chi2,
        "Degrees of Freedom": dof,
        "P-Value": p_value,
        "Cramers V": cramers_v,
        "Minimum Expected Count": minimum_expected,
        "Expected Counts Below 5": expected_below_five,
        "Expected Counts Below 1": expected_below_one,
        "Proportion Expected Below 5": proportion_below_five,
        "Assumptions Met": assumptions_met,
        "Status": "Test completed",
    }

    crosstabs = {
        "counts": observed,
        "row_percentages": percentage_table,
        "expected_counts": expected_table,
    }

    return results, crosstabs


def calculate_phi_matrix(
    data: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate pairwise phi coefficients among binary variables.

    Parameters
    ----------
    data
        DataFrame containing the binary variables.
    variables
        Names of binary predictor columns.

    Returns
    -------
    phi_matrix
        Symmetric matrix of pairwise phi coefficients.
    """

    binary_data = data[list(variables)].copy()

    phi_matrix = binary_data.corr(method="pearson")

    return phi_matrix