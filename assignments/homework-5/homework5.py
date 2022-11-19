import os
import sys
import warnings

# CORRELATION: Referred from 'cat_correlation.py', provided by Professor in class.
# Source: https://teaching.mrsharky.com/code/python/utils/cat_correlation.py
import cat_correlation as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ranking_algos as a4
import seaborn as sns
import sqlalchemy
from plotly import graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


# check and replace nan value with zero for each continuous variable in the data
def replace_nan_value(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and np.isnan(df[col]).any():
            # df[col] = np.fill_na(df[col])
            df[col] = df[col].replace(np.nan, 0)
    return df


# Determine if response type is continuous or boolean
def response_type(df, response):
    if df[response].nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


# Determine if the predictor type is categorical or continuous,
def predictor_type(df, predictor):
    pred_type = []
    cat_cols = []
    cont_cols = []
    for col in predictor:
        if pd.api.types.is_numeric_dtype(df[col]):
            pred_type.append("Continuous")
        else:
            pred_type.append("Categorical")
    for name, p_type in zip(predictor, pred_type):
        if p_type == "Categorical":
            cat_cols.append(name)
        elif p_type == "Continuous":
            cont_cols.append(name)
    return pred_type, cat_cols, cont_cols


# Create Heatmap Plot - by calling its function from Assignment 4
def heatmap_plot(df, col):
    file_path = "plots"
    filename = file_path + "cat_response_cat_predictor_heat_map_" + col + ".html"

    a4.create_heatmap(df, col, filename)
    file_n = "plots/cat_response_cat_predictor_heat_map_" + col + ".html"
    col_link = "<a href=" + file_n + ">" + col

    return col_link


# Create Violin Plot - by calling its function from Assignment 4
def violin_plot(df, col):
    file_path = "plots"
    filename = file_path + "cont_response_cat_predictor_violin_plot_" + col + ".html"

    a4.create_violin(df, col, filename)
    file_n = "plots/cont_response_cat_predictor_violin_plot_" + col + ".html"
    col_link = "<a href=" + file_n + ">" + col

    return col_link


# Create Distribution Plot
# By referring its function from Assignment 4
def distribution_plot(df, col):
    file_path = "plots"
    filename = file_path + "cat_response_cont_predictor_dist_plot_" + col + ".html"

    a4.create_distribution(df, col, filename)
    file_n = "plots/cat_response_cont_predictor_dist_plot_" + col + ".html"
    col_link = "<a href=" + file_n + ">" + col

    return col_link


# Create Scatter Plot - by calling its function from Assignment 4
def scatter_plot(df, col):
    file_path = "plots"
    filename = file_path + "cont_response_cont_predictor_scatter_plot_" + col + ".html"

    a4.create_scatter(df, col, filename)
    file_n = "plots/cont_response_cont_predictor_scatter_plot_" + col + ".html"
    col_link = "<a href=" + file_n + ">" + col

    return col_link


# Calculate Mean Square Difference (MSD) for type Continuous-Continuous
def msd_cont_cont(col1, col2, df):
    bin1 = pd.DataFrame(
        {
            "X1": df[col1],
            "X2": df[col2],
            "Y": df["home_team_wins"],
            # Here, I was getting duplicates into my Bin variables.
            # So, removed the error using ranking method. Reference URL below helped me.
            # https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
            "Bucket1": pd.qcut(df[col1].rank(method="first"), 3),
            "Bucket2": pd.qcut(df[col2].rank(method="first"), 3),
        }
    )
    bin2 = (
        bin1.groupby(["Bucket1", "Bucket2"]).agg({"Y": ["count", "mean"]}).reset_index()
    )
    return bin2


# Calculate Mean Square Difference (MSD) for type Categorical-Continuous
def msd_cat_cont(cat_col, cont_col, df):
    bin1 = pd.DataFrame(
        {
            "X1": df[cat_col],
            "X2": df[cont_col],
            "Y": df["home_team_wins"],
            "Bucket": pd.qcut(df[cont_col].rank(method="first"), 3),
        }
    )
    bin2 = bin1.groupby(["X1", "Bucket"]).agg({"Y": ["count", "mean"]}).reset_index()
    return bin2


# Calculate Mean Square Difference (MSD) for type Weighted-Unweighted
def msd_weighted_unweighted(col1, col2, df, pop_prop_1, res_type):
    if res_type == 3:
        d1_cc = pd.DataFrame(
            {
                "X1": df[col1],
                "X2": df[col2],
                "Y": df["home_team_wins"],
            }
        )
        d2_cc = d1_cc.groupby(["X1", "X2"]).agg({"Y": ["count", "mean"]}).reset_index()

    elif res_type == 2:
        d2_cc = msd_cat_cont(col1, col2, df)
    else:
        d2_cc = msd_cont_cont(col1, col2, df)

    # Find the 'Bincount', 'Binmean' and 'mean of columns'
    d2_cc.columns = [col1, col2, "BinCount", "BinMean"]
    pop_prop = d2_cc.BinCount / len(df)

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

    file_path = "plots"
    filename = file_path + "BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    file_n = "plots/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fname = "<a href=" + file_n + ">Plot Link"
    return d2_cc["Mean_sq_diff"].sum(), d2_cc["Mean_sq_diffW"].sum(), fname


# Calculate Correlation using Pearson's coefficient for type Continuous-Continuous
# Using the result to generate correlation matrix & mean difference in weighted-unweighted
def cont_cont_cor(cont_cols, df, cor_plot):
    df_cols = ["Continuous1", "Continuous2", "Correlation"]
    bf_cols = ["Continuous1", "Continuous2", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cont_cont = pd.DataFrame(columns=bf_cols)
    cont_cont_corr = pd.DataFrame(columns=df_cols)
    cont_cont_matrix = pd.DataFrame(index=cont_cols, columns=cont_cols)
    pop_prop_1 = df.target.sum() / len(df)

    if len(cont_cols) > 1:
        for i in range(len(cont_cols)):
            for j in range(i, len(cont_cols)):
                if cont_cols[i] != cont_cols[j]:
                    val, _ = stats.pearsonr(df[cont_cols[i]], df[cont_cols[j]])

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
                        cont_cols[i], cont_cols[j], df, pop_prop_1, 1
                    )

                    d2_cont_cont = d2_cont_cont.append(
                        dict(zip(bf_cols, [cont_cols[i], cont_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )

                else:
                    cont_cont_matrix[cont_cols[i]][cont_cols[j]] = 1.0
    return cont_cont_corr, cont_cont_matrix, d2_cont_cont


# Calculate Correlation using Pearson's coefficient for type Categorical-Continuous
def cat_cont_cor(cat_cols, cont_cols, df, cor_plot):
    df_cols = ["Categorical", "Continuous", "Correlation"]
    bf_cols = ["Categorical", "Continuous", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cat_cont = pd.DataFrame(columns=bf_cols)
    cat_cont_corr = pd.DataFrame(columns=df_cols)
    cat_cont_matrix = pd.DataFrame(index=cat_cols, columns=cont_cols)
    pop_prop_1 = df.target.sum() / len(df)

    # For correlation, we need atleast one categorical and continuous variable
    if (len(cont_cols) >= 1) and (len(cat_cols) >= 1):
        for i in range(len(cat_cols)):
            for j in range(len(cont_cols)):

                # Used code from "cat_correlation.py" provided by Professor
                val = cc.cat_cont_correlation_ratio(df[cat_cols[i]], df[cont_cols[j]])

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
                    cat_cols[i], cont_cols[j], df, pop_prop_1, 2
                )
                d2_cat_cont = d2_cat_cont.append(
                    dict(zip(bf_cols, [cat_cols[i], cont_cols[j], w, uw, fname])),
                    ignore_index=True,
                )
    return cat_cont_corr, cat_cont_matrix, d2_cat_cont


# Calculate Correlation using Pearson's coefficient for type Categorical-Categorical
def cat_cat_cor(cat_cols, df, cor_plot):
    df_cols = ["Categorical1", "Categorical2", "Correlation"]
    bf_cols = ["Categorical1", "Categorical2", "MeanSqDiff", "MeanSqDiffW", "Plot Link"]
    d2_cat_cat = pd.DataFrame(columns=bf_cols)
    cat_cat_corr = pd.DataFrame(columns=df_cols)
    cat_cat_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    pop_prop_1 = df.target.sum() / len(df)

    # Consider all categorical column combinations
    # NOTE: Ignore the loop if only one categorical column is present
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i, len(cat_cols)):
                if cat_cols[i] != cat_cols[j]:
                    # Dataframe (cat column and target) to calculate difference in mean
                    w, uw, fname = msd_weighted_unweighted(
                        cat_cols[i], cat_cols[j], df, pop_prop_1, 3
                    )

                    d2_cat_cat = d2_cat_cat.append(
                        dict(zip(bf_cols, [cat_cols[i], cat_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )

                    # Cramers-V for correlation
                    val = cc.cat_correlation(df[cat_cols[i]], df[cat_cols[j]])

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


# Main module
def main():
    # Directory for saving all plots
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("bruteforce-plots"):
        os.mkdir("bruteforce-plots")

    # Maria DB Connection Details
    db_database = "baseball"
    db_host = "localhost:3306"
    db_user = "root"
    db_pass = "eldudeH.22"  # pragma: allowlist secret

    # Connection String
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    # sqlalchemy engine
    sql_engine = sqlalchemy.create_engine(connect_string)

    # Data query to load
    query = """
    SELECT * FROM pitcher_stats
    """

    # Data query transformations
    df = pd.read_sql_query(query, sql_engine)
    df = df.replace("", float("NaN"))

    # Mask NaN values by 0 (not excluding them entirely)
    df = replace_nan_value(df)

    df = df.drop(
        labels=[
            "game_id",
            "home_team_id",
            "away_team_id",
            "local_date",
        ],
        axis=1,
    )

    # Check response type
    response = y = "home_team_wins"
    res_type = response_type(df, response)
    print("Response variable is ", res_type)

    # Add all predictors
    predictor = df.columns.tolist()
    predictor.remove(response)
    x = df.drop("home_team_wins", axis=1)

    # PART-1: CORRELATION METRICS
    # Generating Categorical and Continuous columns list
    # Check predictor type
    pred_type, cat_cols, cont_cols = predictor_type(df, predictor)

    print("<<< Categorical Columns >>>\n")
    print(cat_cols)
    print("<<< Continuous Columns >>>")
    print(cont_cols)

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

    # if Categorical then convert values to '1' and '0'
    if res_type == "Boolean":
        df["home_team_wins"] = df["home_team_wins"].astype("category")
        df["home_team_wins"] = df["home_team_wins"].cat.codes
        rf = RandomForestClassifier(
            n_estimators=100, oob_score=True, n_jobs=1, random_state=42
        )
        rf.fit(x, y)
        rf.feature_importances_

    else:
        rf = RandomForestRegressor(random_state=42)
        rf.fit(x, y)
        rf.feature_importances_

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

    # Creating plots for the categorical/continuous values
    col_plot = {}

    for col in cat_cols:
        if res_type == "Boolean":
            col_plot[col] = heatmap_plot(df, col)
        else:
            col_plot[col] = violin_plot(df, col)
    for col in cont_cols:
        if res_type == "Boolean":
            col_plot[col] = distribution_plot(df, col)
        else:
            col_plot[col] = scatter_plot(df, col)

    # Continuous / Continuous type
    cont_cont_corr, cont_cont_matrix, d2_cont_cont = cont_cont_cor(
        cont_cols, df, col_plot
    )
    print("--------------- Continuous-Continuous : Correlation metrics ---------------")
    cont_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cont_cont_corr)

    # Categorical / Continuous type
    cat_cont_corr, cat_cont_matrix, d2_cat_cont = cat_cont_cor(
        cat_cols, cont_cols, df, col_plot
    )
    print(
        "--------------- Categorical-Continuous : Correlation metrics ---------------"
    )
    cat_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cont_corr)

    # Categorical / Categorical type
    cat_cat_corr, cat_cat_matrix, d2_cat_cat = cat_cat_cor(cat_cols, df, col_plot)
    print(
        "--------------- Categorical-Categorical : Correlation metrics ---------------"
    )
    cat_cat_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cat_corr)

    # Collecting all metrics and generating one consolidated html file
    with open("HW5-Part1-Corr-Metric.html", "w") as _file:
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
    fig_1.savefig("plots/cont-cont-corr.png")
    plt.clf()

    # Categorical-Continuous : Correlation Plot
    cat_cont_matrix = cat_cont_matrix.astype(float)
    sns_plot_2 = sns.heatmap(cat_cont_matrix, annot=True)
    fig_2 = sns_plot_2.get_figure()
    fig_2.savefig("plots/cat-cont-corr.png")
    plt.clf()

    # Categorical-Categorical : Correlation Plot
    cat_cat_matrix = cat_cat_matrix.astype(float)
    sns_plot_3 = sns.heatmap(cat_cat_matrix, annot=True)
    fig_3 = sns_plot_3.get_figure()
    fig_3.savefig("plots/cat-cat-corr.png")

    # Collecting all plots and generating one consolidated html file
    with open("HW5-Part2-Corr-Matrices.html", "w") as _file:
        _file.write(
            "<h1> Continuous-Continuous : Correlation Plot </h1> "
            + "<img src='./plots/cont-cont-corr.png'"
            + "alt='Continuous-Continuous Plot'>"
            + "<h1> Categorical-Continuous : Correlation Plot </h1> "
            + "<img src = './plots/cat-cont-corr.png'"
            + "alt='Categorical-Continuous Plot'>"
            + "<h1> Categorical-Categorical : Correlation Plot </h1>"
            + "<img src ='./plots/cat-cat-corr.png'"
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
    with open("HW5-Part3-BruteForce.html", "w") as _file:
        _file.write(
            d2_cont_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cat.to_html(render_links=True, escape=False)
        )

    # Combining all the HTML output files (Part-1,2,3) into one single file
    with open("HW5-Final-Output.html", "w") as _file:
        _file.write(
            "<p><b> MidTerm Output <table><tr>"
            + "<tr><td><a href= 'HW5-Part1-Corr-Metric.html'>"
            + "1. Correlation Metrics"
            + "<tr><td> <a href= 'HW5-Part2-Corr-Matrices.html'>"
            + "2. Correlation Plot"
            + "<tr><td> <a href= 'HW5-Part3-BruteForce.html'>"
            + "3. Brute-Force"
        )

    # 1st Model: Logistic Regression
    x_train, x_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1].values,
        df["home_team_wins"],
        test_size=0.25,
        # shuffle=False,
        random_state=42,
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    logic_predict = lr.predict(x_test)
    lr_accuracy = accuracy_score(y_test, logic_predict)  # accuracy
    print("(1) Logistic Regression Model: ")
    print("Accuracy is : ", round(lr_accuracy, 2), "\n")

    # 2nd Model: Random Forest Classifier
    rfc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rfc", RandomForestClassifier(random_state=1234)),
        ]
    )
    rfc.fit(x_train, y_train)
    rfc_predict = rfc.predict(x_test)
    rfc_accuracy = accuracy_score(y_test, rfc_predict)
    print("(2) Random Forest Model: ")
    print("Accuracy is : ", round(rfc_accuracy, 2), "\n")

    # 3rd Model: Naive Bayes
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    nb_predict = nb.predict(x_test)
    nb_accuracy = accuracy_score(y_test, nb_predict)  # accuracy
    print("(3) Naive Bayes: ")
    print("Accuracy is : ", round(nb_accuracy, 2), "\n")

    # 4th Model: Neural Networks (NN)
    nn = MLPClassifier()
    nn.fit(x_train, y_train)
    nn_predict = nn.predict(x_test)
    nn_accuracy = accuracy_score(y_test, nn_predict)
    print("Neural Network (NN) Accuracy:", round(nn_accuracy, 2), "\n")

    # 5th Model: K Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(35)
    knn.fit(x_train, y_train)
    knn_predict = knn.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_predict)
    print("K Nearest Neighbors (KNN) Accuracy:", round(knn_accuracy, 2), "\n\n")

    # Final Result for best performing model
    print("Comparison of five ML Model's performances: ")

    best_model = {
        lr_accuracy: "Logistic Regression",
        # rfc_accuracy: "Random Forest Classifier",
        nb_accuracy: "Naive Bayes",
        nn_accuracy: "Neural Networks (NN)",
        knn_accuracy: "K Nearest Neighbors (KNN)",
    }

    print(
        best_model.get(max(best_model)),
        " is the better performing model from all the five models.",
    )


# Main() check
if __name__ == "__main__":
    sys.exit(main())
