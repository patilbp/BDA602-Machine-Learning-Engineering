import os
import sys
import warnings

# CORRELATION: Referred from 'cat_correlation.py', provided by Professor in class.
# Source: https://teaching.mrsharky.com/code/python/utils/cat_correlation.py
import cat_correlation as cc
import matplotlib.pyplot as plt
import pandas as pd

# I've called plot functions from Homework-4, and used them here.
import ranking_algos as a4
import seaborn as sns
from plotly import graph_objects as go
from scipy import stats

warnings.simplefilter(action="ignore", category=FutureWarning)


# Default data I've passed here is the "seaborn dataset - mpg"
# NOTE: If you want to change the dataset, please edit here.
# I could not integrate the 'dataset_loader.py' file provided by Professor in the class.
# Hence, I tried this workaround.


def default_data():
    data_set = sns.load_dataset(name="mpg").dropna().reset_index()
    # I'm unable to remove the pre-commit error for the predictors
    # Reason being, it is not getting called in the 'return'
    predictors = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "origin",
        "name",
    ]
    response = "mpg"

    print(f"Default Data set selected: {response}")
    return data_set


# Load Data using only the response name in form of string.
def load_data(file_name, res):
    # Validate if any file is passed
    if file_name == "":
        # NOTE: modify the response name here, if you've changed the data source
        res = "mpg"
        try:
            data_set = default_data()
        except FileNotFoundError:
            data_set = default_data()

        # Change the response variable generic to use as target column
        data_set = data_set.rename(columns={res: "target"})

    # Replace any NULL values by '0', instead of removing them
    data_set = data_set.fillna(0)
    data_set.reset_index(drop=True, inplace=True)
    return data_set


# Determine if response type is continuous or boolean
def response_type(df):
    if df.target.nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


# Determine if the predictor type is categorical or continuous,
def predictor_type(pred):
    if pred.dtypes.name in ["category", "object"] or pred.nunique() < 5:
        return True
    else:
        return False


# Generate all categorical-continuous columns
def get_cat_cont_columns(x, data_set):
    cat_cols = []
    cont_cols = []
    for col in x.columns:
        col_name = col.replace(" ", "-").replace("/", "-")

        if predictor_type(x[col]):
            data_set[col] = data_set[col].astype("category")
            data_set[col] = data_set[col].cat.codes
            cat_cols.append(col_name)
            # I was getting Future Warnings for .append, as it is going to deprecate
            # So, I tried using .concat, but it was not helpful. Hence, commented.
            # cat_cols = pd.concat(cat_cols)

        else:
            cont_cols.append(col_name)
            # cont_cols = pd.concat(cont_cols)

    return cat_cols, cont_cols


# Create Heatmap Plot - by calling its function from Assignment 4
def heatmap_plot(data_set, col):
    filename = file_path + "cat_response_cat_predictor_heat_map_" + col + ".html"

    a4.create_heatmap(data_set, col, filename)
    file_n = "midterm_plots/cat_response_cat_predictor_heat_map_" + col + ".html"
    col_1 = "<a href=" + file_n + ">" + col

    return col_1


# Create Violin Plot - by calling its function from Assignment 4
def violin_plot(data_set, col):
    filename = file_path + "cont_response_cat_predictor_violin_plot_" + col + ".html"

    a4.create_violin(data_set, col, filename)
    file_n = "midterm_plots/cont_response_cat_predictor_violin_plot_" + col + ".html"
    col_1 = "<a href=" + file_n + ">" + col

    return col_1


# Create Distribution Plot
# By referring its function from Assignment 4
def distribution_plot(data_set, col):
    filename = file_path + "cat_response_cont_predictor_dist_plot_" + col + ".html"
    a4.create_distribution(data_set, col, filename)
    file_n = "midterm_plots/cat_response_cont_predictor_dist_plot_" + col + ".html"
    col_1 = "<a href=" + file_n + ">" + col
    return col_1


# Create Scatter Plot - by calling its function from Assignment 4
def scatter_plot(data_set, col):
    filename = file_path + "cont_response_cont_predictor_scatter_plot_" + col + ".html"
    a4.create_scatter(data_set, col, filename)
    file_n = "midterm_plots/cont_response_cont_predictor_scatter_plot_" + col + ".html"
    col_1 = "<a href=" + file_n + ">" + col
    return col_1


# Calculate Mean Square Difference (MSD) for type Continuous-Continuous
def msd_cont_cont(col1, col2, data_set):
    bin1 = pd.DataFrame(
        {
            "X1": data_set[col1],
            "X2": data_set[col2],
            "Y": data_set["target"],
            # Here, I was getting duplicates into my Bin variables.
            # So, removed the error using ranking method. Reference URL below helped me.
            # https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
            "Bucket1": pd.qcut(data_set[col1].rank(method="first"), 3),
            "Bucket2": pd.qcut(data_set[col2].rank(method="first"), 3),
        }
    )
    bin2 = (
        bin1.groupby(["Bucket1", "Bucket2"]).agg({"Y": ["count", "mean"]}).reset_index()
    )
    return bin2


# Calculate Mean Square Difference (MSD) for type Categorical-Continuous
def msd_cat_cont(cat_col, cont_col, data_set):
    bin1 = pd.DataFrame(
        {
            "X1": data_set[cat_col],
            "X2": data_set[cont_col],
            "Y": data_set["target"],
            "Bucket": pd.qcut(data_set[cont_col].rank(method="first"), 3),
        }
    )
    bin2 = bin1.groupby(["X1", "Bucket"]).agg({"Y": ["count", "mean"]}).reset_index()
    return bin2


# Calculate Mean Square Difference (MSD) for type Weighted-Unweighted
def msd_weighted_unweighted(col1, col2, data_set, pop_prop_1, res_type):
    if res_type == 3:
        d1_cc = pd.DataFrame(
            {
                "X1": data_set[col1],
                "X2": data_set[col2],
                "Y": data_set["target"],
            }
        )
        d2_cc = d1_cc.groupby(["X1", "X2"]).agg({"Y": ["count", "mean"]}).reset_index()

    elif res_type == 2:
        d2_cc = msd_cat_cont(col1, col2, data_set)
    else:
        d2_cc = msd_cont_cont(col1, col2, data_set)

    # Find the 'Bincount', 'Binmean' and 'mean of columns'
    d2_cc.columns = [col1, col2, "BinCount", "BinMean"]
    pop_prop = d2_cc.BinCount / len(data_set)

    # Find MeansqDiff for weighted and unweighted type
    d2_cc["Mean_sq_diff"] = (d2_cc["BinMean"] - pop_prop_1) ** 2
    d2_cc["Mean_sq_diffW"] = d2_cc.Mean_sq_diff * pop_prop

    # Generate Mean Square Difference - Plot
    d_mat = d2_cc.pivot(index=col1, columns=col2, values="Mean_sq_diffW")
    fig = go.Figure(data=[go.Surface(z=d_mat.values)])
    fig.update_layout(
        title=col1 + " " + col2 + " Plot",
        autosize=True,
        scene=dict(xaxis_title=col2, yaxis_title=col1, zaxis_title="target"),
    )

    filename = file_path + "BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    file_n = "midterm_plots/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fname = "<a href=" + file_n + ">Plot Link"
    return d2_cc["Mean_sq_diff"].sum(), d2_cc["Mean_sq_diffW"].sum(), fname


# Calculate Correlation using Pearson's coefficient for type Continuous-Continuous
# Using the result to generate correlation matrix & mean difference in weighted-unweighted
def cont_cont_cor(cont_cols, data_set, cor_plot):
    df_cols = ["Continuous1", "Continuous2", "Correlation"]
    bf_cols = ["Continuous1", "Continuous2", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cont_cont = pd.DataFrame(columns=bf_cols)
    cont_cont_corr = pd.DataFrame(columns=df_cols)
    cont_cont_matrix = pd.DataFrame(index=cont_cols, columns=cont_cols)
    pop_prop_1 = data_set.target.sum() / len(data_set)

    if len(cont_cols) > 1:
        for i in range(len(cont_cols)):
            for j in range(i, len(cont_cols)):
                if cont_cols[i] != cont_cols[j]:
                    val, _ = stats.pearsonr(
                        data_set[cont_cols[i]], data_set[cont_cols[j]]
                    )

                    cont_cont_matrix.loc[cont_cols[i]][cont_cols[j]] = val
                    cont_cont_matrix.loc[cont_cols[j]][cont_cols[i]] = val
                    cont_cont_corr = cont_cont_corr.append(
                        dict(
                            zip(
                                df_cols,
                                [cor_plot[cont_cols[i]], cor_plot[cont_cols[j]], val],
                            )
                        ),
                        ignore_index=True,
                    )

                    w, uw, fname = msd_weighted_unweighted(
                        cont_cols[i], cont_cols[j], data_set, pop_prop_1, 1
                    )

                    d2_cont_cont = d2_cont_cont.append(
                        dict(zip(bf_cols, [cont_cols[i], cont_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )

                else:
                    cont_cont_matrix[cont_cols[i]][cont_cols[j]] = 1.0
    return cont_cont_corr, cont_cont_matrix, d2_cont_cont


# Calculate Correlation using Pearson's coefficient for type Categorical-Continuous
def cat_cont_cor(cat_cols, cont_cols, data_set, cor_plot):
    df_cols = ["Categorical", "Continuous", "Correlation"]
    bf_cols = ["Categorical", "Continuous", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cat_cont = pd.DataFrame(columns=bf_cols)
    cat_cont_corr = pd.DataFrame(columns=df_cols)
    cat_cont_matrix = pd.DataFrame(index=cat_cols, columns=cont_cols)
    pop_prop_1 = data_set.target.sum() / len(data_set)

    # For correlation, we need atleast one categorical and continuous variable
    if (len(cont_cols) >= 1) and (len(cat_cols) >= 1):
        for i in range(len(cat_cols)):
            for j in range(len(cont_cols)):

                # Used code from "cat_correlation.py" provided by Professor
                val = cc.cat_cont_correlation_ratio(
                    data_set[cat_cols[i]], data_set[cont_cols[j]]
                )

                cat_cont_corr = cat_cont_corr.append(
                    dict(
                        zip(
                            df_cols,
                            [cor_plot[cat_cols[i]], cor_plot[cont_cols[j]], val],
                        )
                    ),
                    ignore_index=True,
                )
                cat_cont_matrix.loc[cat_cols[i]][cont_cols[j]] = val
                w, uw, fname = msd_weighted_unweighted(
                    cat_cols[i], cont_cols[j], data_set, pop_prop_1, 2
                )
                d2_cat_cont = d2_cat_cont.append(
                    dict(zip(bf_cols, [cat_cols[i], cont_cols[j], w, uw, fname])),
                    ignore_index=True,
                )
    return cat_cont_corr, cat_cont_matrix, d2_cat_cont


# Calculate Correlation using Pearson's coefficient for type Categorical-Categorical
def cat_cat_cor(cat_cols, data_set, cor_plot):
    df_cols = ["Categorical1", "Categorical2", "Correlation"]
    bf_cols = ["Categorical1", "Categorical2", "MeanSqDiff", "MeanSqDiffW", "Plot Link"]
    d2_cat_cat = pd.DataFrame(columns=bf_cols)
    cat_cat_corr = pd.DataFrame(columns=df_cols)
    cat_cat_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    pop_prop_1 = data_set.target.sum() / len(data_set)

    # Consider all categorical column combinations
    # NOTE: Ignore the loop if only one categorical column is present
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i, len(cat_cols)):
                if cat_cols[i] != cat_cols[j]:
                    # Dataframe (cat column and target) to calculate difference in mean
                    w, uw, fname = msd_weighted_unweighted(
                        cat_cols[i], cat_cols[j], data_set, pop_prop_1, 3
                    )

                    d2_cat_cat = d2_cat_cat.append(
                        dict(zip(bf_cols, [cat_cols[i], cat_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )

                    # Cramers-V for correlation
                    val = cc.cat_correlation(
                        data_set[cat_cols[i]], data_set[cat_cols[j]]
                    )

                    cat_cat_corr = cat_cat_corr.append(
                        dict(
                            zip(
                                df_cols,
                                [cor_plot[cat_cols[i]], cor_plot[cat_cols[j]], val],
                            )
                        ),
                        ignore_index=True,
                    )

                    # Plot the correlation matrix
                    cat_cat_matrix.loc[cat_cols[i]][cat_cols[j]] = val
                    cat_cat_matrix.loc[cat_cols[j]][cat_cols[i]] = val

                else:
                    cat_cat_matrix.loc[cat_cols[i]][cat_cols[j]] = 1
    return cat_cat_corr, cat_cat_matrix, d2_cat_cat


# Assigning directory path for saving all midterm plots
file_path = "~/midterm_plots/"


def main(file_name, response):
    data_df = load_data(file_name, response)
    print(data_df.head())
    x = data_df.drop("target", axis=1)

    res_type = response_type(data_df)
    print("Response variable is " + res_type)

    if res_type == "Boolean":
        data_df["target"] = data_df["target"].astype("category")
        data_df["target"] = data_df["target"].cat.codes

    # creating directory for saving all midterm plots
    if not os.path.exists("~/midterm_plots"):
        print("Creating plots")
        os.makedirs("~/midterm_plots")

    # file_path = "~/midterm_plots/"

    # PART-1: CORRELATION METRICS
    # Generating Categorical and Continuous columns list
    cat_cols, cont_cols = get_cat_cont_columns(x, data_df)
    print("--------------- Categorical Columns ---------------")
    print(cat_cols)
    print("--------------- Continuous Columns ---------------")
    print(cont_cols)
    col_plot = {}

    for col in cat_cols:
        if res_type == "Boolean":
            col_plot[col] = heatmap_plot(data_df, col)
        else:
            col_plot[col] = violin_plot(data_df, col)
    for col in cont_cols:
        if res_type == "Boolean":
            col_plot[col] = distribution_plot(data_df, col)
        else:
            col_plot[col] = scatter_plot(data_df, col)

    # Continuous / Continuous type
    cont_cont_corr, cont_cont_matrix, d2_cont_cont = cont_cont_cor(
        cont_cols, data_df, col_plot
    )
    print("--------------- Continuous-Continuous : Correlation metrics ---------------")
    cont_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cont_cont_corr)

    # Categorical / Continuous type
    cat_cont_corr, cat_cont_matrix, d2_cat_cont = cat_cont_cor(
        cat_cols, cont_cols, data_df, col_plot
    )
    print(
        "--------------- Categorical-Continuous : Correlation metrics ---------------"
    )
    cat_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cont_corr)

    # Categorical / Categorical type
    cat_cat_corr, cat_cat_matrix, d2_cat_cat = cat_cat_cor(cat_cols, data_df, col_plot)
    print(
        "--------------- Categorical-Categorical : Correlation metrics ---------------"
    )
    cat_cat_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cat_corr)

    # Collecting all metrics and generating one consolidated html file
    with open("~/Midterm-Part1-Corr-Metric.html", "w") as _file:
        _file.write(
            cont_cont_corr.to_html(render_links=True, escape=False)
            + "<br>"
            + cat_cont_corr.to_html(render_links=True, escape=False)
            + "<br>"
            + cat_cat_corr.to_html(render_links=True, escape=False)
        )

    # PART-2: CORRELATION MATRICES
    # Continuous-Continuous : Correlation Plot
    cont_cont_matrix = cont_cont_matrix.astype(float)
    sns_plot_1 = sns.heatmap(cont_cont_matrix, annot=True)
    fig_1 = sns_plot_1.get_figure()
    fig_1.savefig("~/midterm_plots/cont-cont-corr.png")
    plt.clf()

    # Categorical-Continuous : Correlation Plot
    cat_cont_matrix = cat_cont_matrix.astype(float)
    sns_plot_2 = sns.heatmap(cat_cont_matrix, annot=True)
    fig_2 = sns_plot_2.get_figure()
    fig_2.savefig("~/midterm_plots/cat-cont-corr.png")
    plt.clf()

    # Categorical-Categorical : Correlation Plot
    cat_cat_matrix = cat_cat_matrix.astype(float)
    sns_plot_3 = sns.heatmap(cat_cat_matrix, annot=True)
    fig_3 = sns_plot_3.get_figure()
    fig_3.savefig("~/midterm_plots/cat-cat-corr.png")

    # Collecting all plots and generating one consolidated html file
    with open("~/Midterm-Part2-Corr-Matrices.html", "w") as _file:
        _file.write(
            "<h1> Continuous-Continuous : Correlation Plot </h1> "
            + "<img src='./midterm_plots/cont-cont-corr.png'"
            + "alt='Continuous-Continuous Plot'>"
            + "<h1> Categorical-Continuous : Correlation Plot </h1> "
            + "<img src = './midterm_plots/cat-cont-corr.png'"
            + "alt='Categorical-Continuous Plot'>"
            + "<h1> Categorical-Categorical : Correlation Plot </h1>"
            + "<img src ='./midterm_plots/cat-cat-corr.png'"
            + "alt='Categorical-Categorical'>"
        )

    # PART-3 : BRUTE FORCE
    # Continuous-Continuous : Difference of mean
    print("--------------- Continuous-Continuous : Brute Force ---------------")
    d2_cont_cont = d2_cont_cont.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cont_cont)

    # Categorical-Continuous : Difference of mean
    print("--------------- Categorical-Continuous : Brute Force ---------------")
    d2_cat_cont = d2_cat_cont.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cat_cont)

    # Categorical-Categorical : Difference of mean
    print("--------------- Categorical-Categorical : Brute Force ---------------")
    d2_cat_cat = d2_cat_cat.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cat_cat)

    # Collecting all tables and generating one consolidated html file
    with open("~/Midterm-Part3-BruteForce.html", "w") as _file:
        _file.write(
            d2_cont_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cat.to_html(render_links=True, escape=False)
        )

    # Combining all the HTML output files (Part-1,2,3) into one single file
    with open("~/Midterm-Final-Output.html", "w") as _file:
        _file.write(
            "<p><b> MidTerm Output <table><tr>"
            + "<tr><td><a href= 'Midterm-Part1-Corr-Metric.html'>"
            + "1. Correlation Metrics"
            + "<tr><td> <a href= 'Midterm-Part2-Corr-Matrices.html'>"
            + "2. Correlation Plot"
            + "<tr><td> <a href= 'Midterm-Part3-BruteForce.html'>"
            + "3. Brute-Force"
        )


# Main() check
if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else ""
    response = sys.argv[2] if len(sys.argv) > 1 else ""
    sys.exit(main(file, response))
