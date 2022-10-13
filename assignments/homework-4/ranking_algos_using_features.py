import os
import sys

import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
import statsmodels.api as sapi
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix


# Validate if Response type is Categorical or Continuous
def response_type(dataset_df):
    if dataset_df.target.nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


# Validate if Predictor type is Categorical, as categorical type has less unique values
def predictor_type(predictor_field):
    if (
        predictor_field.dtypes == "object"
        or (predictor_field.nunique() / predictor_field.count()) < 0.05
    ):
        return True
    else:
        return False


# Create Heatmap plot for 'Categorical Predictor by Categorical Response'
def create_heatmap(df, col, filename):
    confusion_mat = confusion_matrix(df[col], df["target"])

    # design heatmap plot
    heat_fig = go.Figure(
        data=go.Heatmap(z=confusion_mat, zmin=0, zmax=confusion_mat.max())
    )

    heat_fig.update_layout(
        title="Heatmap Plot for - Categorical Predictor by Categorical Response ",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )

    # display Heatmap plot
    heat_fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


# Create Violin plot for 'Continuous Response by Categorical Predictor'
def create_violin(df, col, filename):
    group_names = pd.unique(df[col])

    # design violin plot
    violin_fig = go.Figure()

    # adding separate axis titles
    for i in group_names:
        y_title = df["target"][df[col] == i]
        x_title = df[col][df[col] == i]

        violin_fig.add_trace(
            go.Violin(
                y=y_title,
                x=x_title,
                name=i.astype(str)
                # box_visible=True,
                # meanline_visible=True
            )
        )

    violin_fig.update_layout(
        title="Violin Plot for - Continuous Response by Categorical Predictor",
        xaxis_title="Group Names",
        yaxis_title="Response",
    )

    # display Violin plot
    violin_fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


# Create Distribution plot for 'Continuous Response by Categorical Predictor'
def create_distribution(df, col, filename):
    group_labels = ["0", "1"]

    # adding custom bin-sizes for distribution plot
    x1 = df[df["target"] == 0][col]
    x3 = df[df["target"] == 1][col]
    hist_data = [x1, x3]

    # design distribution plot
    distribution_fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2)

    distribution_fig.update_layout(
        title="Distribution Plot for - Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )

    # display Distribution plot
    distribution_fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


# Create Scatter plot for 'Continuous Response by Categorical Predictor'
def create_scatter(df, col, filename):
    # Hint by Professor: Use OLS for Continuous Response
    scatter_fig = px.scatter(x=df[col], y=df["target"], trendline="ols")

    scatter_fig.update_layout(
        title="Scatter Plot for - Continuous Response by Continuous Predictor ",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )

    # display Scatter plot
    scatter_fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


# Create 'Difference with mean of response plot'
def create_diff_mean_response(df, pop_divide):
    diff_mean_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # design difference with mean response plot
    diff_mean_fig.add_trace(
        go.Bar(x=df["Mean"], y=df["COUNT"], name="Population"),
        secondary_y=False,
    )

    diff_mean_fig.add_trace(
        go.Scatter(
            x=df["Mean"], y=df["BinMean"], line=dict(color="red"), name="Mean of Bin"
        ),
        secondary_y=True,
    )

    diff_mean_fig.add_trace(
        go.Scatter(
            x=df["Mean"],
            y=df["Pop_mean"],
            line=dict(color="green"),
            name="Mean of Population",
        ),
        secondary_y=True,
    )

    diff_mean_fig.update_layout(
        height=600,
        width=800,
        title_text="Binned Difference with Mean of Response vs Bin",
    )
    filename = "~/plots/Diff_in_Mean_with_Response.html"

    diff_mean_fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


def mean_unweighted_weighted(d2, pop_divide, df):
    d3 = pd.DataFrame({}, index=[])
    d3["Mean"] = d2.mean().X
    d3["LowerBin"] = d2.min().X
    d3["UpperBin"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["Pop_mean"] = pop_divide
    proportion = d3["COUNT"] / len(df)
    d3["BinMean"] = d2.mean().Y
    d3["Mean_sq_diff"] = (d3.BinMean - pop_divide) ** 2
    d3["Mean_sq_diffW"] = d3.Mean_sq_diff * proportion

    create_diff_mean_response(d3, pop_divide)
    return d3["Mean_sq_diff"].sum(), d3["Mean_sq_diffW"].sum()


# Add - Bin Candidate Predictor Variable
# Reference: https://teaching.mrsharky.com/sdsu_fall_2020_lecture06.html#/6/0/10

max_bin = 10
force_bin = 3


def mean_of_response_cont(df, col, pop_divide, n=max_bin):
    print(col)
    r = 0

    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame(
                {"X": df[col], "Y": df["target"], "Bucket": pd.qcut(df[col], n)}
            )

            d2 = d1.groupby("Bucket", as_index=True)

            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1

        except Exception:
            n = n - 1

    # Working on quantile part. Pending as of now.
    # Creating bins
    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(df[col], np.linspace(0, 1, n))

        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)

        d1 = pd.DataFrame(
            {
                "X": df[col],
                "Y": df["target"],
                "Bucket": pd.cut(df[col], np.unique(bins), include_lowest=True),
            }
        )

        d2 = d1.groupby("Bucket", as_index=True)

    return mean_unweighted_weighted(d2, pop_divide, df)


def mean_of_response_cat(df, col, pop_divide):
    d1 = pd.DataFrame({"X": df[col], "Y": df["target"]}).groupby(df[col])
    return mean_unweighted_weighted(d1, pop_divide, df)


# Reference UCI Website:
# https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
def read_breast_cancer_data():

    data_url = "https://raw.githubusercontent.com/patilbp/datasets/main/wisconsin-breast-cancer.csv"
    columns = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]

    used_dataset = pd.read_csv(data_url, names=columns)
    return used_dataset


def main(file, response):
    # finds if any csv is passed as input
    if file == "1":
        if response == "1":
            data = load_boston()
            res_name = "Species"

        elif response == "2":
            data = load_diabetes()
            res_name = "Outcome"

        else:
            data = load_breast_cancer()
            res_name = "diagnosis"

        read_data = pd.DataFrame(data.data, columns=data.feature_names)
        read_data["target"] = pd.Series(data.target)

    # if not found any dataset then read dataset of Breast Cancer.CSV
    else:
        if file == "":
            read_data = read_breast_cancer_data()
        else:
            try:
                read_data = pd.read_csv(file)
            except FileNotFoundError:
                read_data = read_breast_cancer_data()

        # if user gives no response variable then use the Breast Cancer response variable
        if response == "":
            response = "diagnosis"

        res_name = response
        print(response)

    # make the column names generic, by using target column as response variable
    read_data = read_data.rename(columns={response: "target"})
    print(read_data.head())

    # get rid of NULL Values
    read_data = read_data.dropna(axis=1, how="any")

    y = read_data.target.values
    x = read_data.drop("target", axis=1)

    # check if directory exists for saving plots, if not then create one
    if not os.path.exists("~/plots"):
        print("Creating plots")
        os.makedirs("~/plots")
    file_path = "~/plots/"

    # resultant tabular data frame - column names
    col_names = [
        "Response",
        "Response_Type",
        "Predictor",
        "Cat/Con",
        "Plot_Link",
        "t-value",
        "p-value",
        "m-plot",
        "RandomForestVarImp",
        "MeanSqDiff",
        "MeanSqDiffWeighted",
        "MeanSqDiffPlots",
    ]
    output_df = pd.DataFrame(columns=col_names)

    # validate if Response variable type is 'Boolean' or 'Continuous'
    res_type = response_type(read_data)
    print("Response variable is: " + res_type)

    # if Categorical then convert values to '1' and '0'
    if res_type == "Boolean":
        read_data["target"] = read_data["target"].astype("category")
        read_data["target"] = read_data["target"].cat.codes
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=1)
        rf.fit(x, y)
        importance = rf.feature_importances_

    else:
        rf = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=1)
        rf.fit(x, y)
        importance = rf.feature_importances_

    # resultant table - metric values
    output_df["Predictor"] = x.columns
    out = []
    f_path = []
    p_val = []
    t_val = []
    m_plot = []
    mean_sq_diff_unweighted = []
    mean_sq_diff_weighted = []
    plot_lk = []

    # replace spaces and slashes by hyphens
    for col in x.columns:
        col_name = col.replace(" ", "-").replace("/", "-")

        # if 'categorical response - categorical predictor', then create heatmap plot
        if predictor_type(x[col]) and res_type == "Boolean":
            out.append("Categorical")
            filename = (
                file_path + "cat_response_cat_predictor_heat_map_" + col_name + ".html"
            )

            create_heatmap(read_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")

        # if 'continuous response - categorical predictor', then create violin plot
        elif predictor_type(x[col]) and res_type != "Boolean":
            out.append("Categorical")
            filename = (
                file_path
                + "cont_response_cat_predictor_violin_plot_"
                + col_name
                + ".html"
            )

            create_violin(read_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")

        # if 'categorical response - continuous predictor', then create distribution plot
        elif not (predictor_type(x[col])) and res_type == "Boolean":
            out.append("Continuous")
            filename = (
                file_path
                + "cat_response_cont_predictor_dist_plot_"
                + col_name
                + ".html"
            )

            create_distribution(read_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")

        # if 'continuous response - continuous predictor', then create scatter plot
        else:
            out.append("Continuous")
            filename = (
                file_path
                + "cont_response_cont_predictor_scatter_plot_"
                + col_name
                + ".html"
            )

            create_scatter(read_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")

        # adding the calculation of 'p-value' and 't-score'
        if res_type == "Boolean":
            predictor = sapi.add_constant(read_data[col])

            # add logistic regression on data
            logit = sapi.Logit(read_data["target"], predictor)
            logit_fitted = logit.fit()

            # calculate values
            t_value = round(logit_fitted.tvalues[1], 6)
            t_val.append(t_value)

            p_value = "{:.6e}".format(logit_fitted.pvalues[1])
            p_val.append(p_value)

            # create scatter plot
            fig = px.scatter(x=read_data[col], y=read_data["target"], trendline="ols")

            fig.update_layout(
                title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {col}",
                yaxis_title="y",
            )

            # display scatter plot
            fig.write_html(
                file=f"~/plots/ranking_{col_name}.html", include_plotlyjs="cdn"
            )

            # generating link to the plot
            filename = "~/plots/ranking_" + col_name + ".html"
            m_plot.append("<a href=" + filename + ">" + filename + "</a>")

        else:
            predictor = sapi.add_constant(read_data[col])

            # add linear regression on data
            linear_regression_model = sapi.OLS(read_data["target"], predictor)
            linear_regression_model_fitted = linear_regression_model.fit()

            # calculate values
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            t_val.append(t_value)
            p_val.append(p_value)

            # create scatter plot
            fig = px.scatter(x=read_data[col], y=read_data["target"], trendline="ols")

            fig.update_layout(
                title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {col}",
                yaxis_title="y",
            )

            # display scatter plot
            fig.write_html(
                file=f"{file_path}/ranking_{col_name}.html", include_plotlyjs="cdn"
            )

            # generating link to the plot
            filename = "~/plots/ranking_" + col_name + ".html"
            m_plot.append("<a href=" + filename + ">" + filename + "</a>")

        # calculating the 'Mean Squared Difference' values by weighted/un-weighted
        if predictor_type(x[col]):
            pop_divide = read_data.target.sum() / len(read_data)
            un_weighted, weighted = mean_of_response_cat(read_data, col, pop_divide)

        else:
            pop_divide = read_data.target.sum() / len(read_data)
            un_weighted, weighted = mean_of_response_cont(read_data, col, pop_divide)

        mean_sq_diff_unweighted.append(un_weighted)
        mean_sq_diff_weighted.append(weighted)

        plot_lk.append(
            "<a href= ~/plots/Diff_in_mean_with_response.html> Plot Link </a>"
        )

    # mapping resultant tabular data frame by row values
    output_df["Response"] = res_name
    output_df["Response_Type"] = res_type
    output_df["Cat/Con"] = out
    output_df["Plot_Link"] = f_path
    output_df["t-value"] = t_val
    output_df["p-value"] = p_val
    output_df["m-plot"] = m_plot
    output_df["RandomForestVarImp"] = importance
    output_df["MeanSqDiff"] = mean_sq_diff_unweighted
    output_df["MeanSqDiffWeighted"] = mean_sq_diff_weighted
    output_df["MeanSqDiffPlots"] = plot_lk

    # Display final resultant table with all variables and their rankings
    output_df.to_html(
        "ranking_algos_using_features.html", render_links=True, escape=False
    )


if __name__ == "__main__":
    f_name = sys.argv[1] if len(sys.argv) > 1 else ""
    # print(f_name)
    r_name = sys.argv[2] if len(sys.argv) > 1 else ""
    # print(r_name)
    sys.exit(main(f_name, r_name))
