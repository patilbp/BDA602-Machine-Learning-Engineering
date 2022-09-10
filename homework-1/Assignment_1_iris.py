import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Iris dataset URL
iris_data_url = "https://teaching.mrsharky.com/data/iris.data"

# Assigning the attribute names as column names
iris_column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class_type",
]
# -----------------------------------------------------------------------------------------------------

# (A) Function for loading iris data into a pandas data-frame
# Passing the data URL as function argument and converting it to pandas data-frame


def data_loader(path=iris_data_url) -> pd.DataFrame:
    df = pd.read_csv(path, names=iris_column_names)
    return df


# Calling the main() function with respective sub-functions
def main() -> int:
    df_iris = data_loader()
    print("\n\n########## Iris Data Frame ##########\n")
    print(df_iris.head())

    # Passing the loaded data-frame as function argument and converting it to pandas data-frame
    df_iris = df_iris.dropna()

    # Displaying basic info of the data-frame
    print("\n\n\n########## Iris data-set basic information ##########\n")
    df_iris.info()

    # Passing the cleaned data-frame as function argument and converting it to numpy array,
    # for performing Statistical Operations on the numpy array generated.
    iris_np = np.array(df_iris)

    # -----------------------------------------------------------------------------------------------------

    # (B) Data-frame Summary Statistics

    # I took reference of .describe() function of pandas, and calculated in numpy accordingly.
    # For instance, df.describe() -> gives us all details of the basic statistical summary of data

    print("\n\n\n ########## Statistics of Summary ##########\n")
    print(f"Shape: {np.shape(iris_np)}")
    print(f"Minimum Value: {np.min(iris_np[:, :-1], axis=0)}")
    print(f"Maximum Value: {np.max(iris_np[:, :-1], axis=0)}")
    print(f"Mean Value: {np.mean(iris_np[:, :-1], axis=0)}")
    print(f"Standard Deviation: { np.std(iris_np[:, :-1]) }")
    print(f"Quantile - 25 %: {np.quantile(iris_np[:, :-1], 0.25, axis=0)}")
    print(f"Quantile - 50 %: {np.quantile(iris_np[:, :-1], 0.50, axis=0)}")
    print(f"Quantile - 75 %: {np.quantile(iris_np[:, :-1], 0.75, axis=0)}")

    # -----------------------------------------------------------------------------------------------------

    # (C)  Plotting data into visuals using plotly

    # (1) SCATTER plots:
    # (i) Sepal-width vs Sepal-length:
    figure = px.scatter(
        df_iris,
        x="sepal_width",
        y="sepal_length",
        color="class_type",
        title="Scatter plot: Sepal-width vs Sepal-length",
    )
    figure.show()

    # (ii) Petal-width vs Petal-length:
    figure = px.scatter(
        df_iris,
        x="petal_width",
        y="petal_length",
        color="class_type",
        title="Scatter plot: Petal-width vs Petal-length",
    )
    figure.show()

    # (iii) Sepal-width vs Sepal-length - with Petal-length as Size and Petal-width into details:

    # I tried searching the relation between sepal and petal and then came across this link below:
    # Reference link : https://qr.ae/pvOMtT

    figure = px.scatter(
        df_iris,
        x="sepal_width",
        y="sepal_length",
        size="petal_length",
        hover_data=["petal_width"],
        color="class_type",
        title="Scatter plot: Sepal-width vs Sepal-length",
    )
    figure.show()

    # (2) VIOLIN plots:
    # (i) Sepal-width:
    figure = px.violin(
        df_iris,
        x="class_type",
        y="sepal_width",
        color="class_type",
        box=True,
        points="all",
        hover_data=df_iris.columns,
        title="Violin plot: Sepal-width as attribute",
    )
    figure.show()

    # (ii) Sepal-length:
    figure = px.violin(
        df_iris,
        x="class_type",
        y="sepal_length",
        color="class_type",
        box=True,
        points="all",
        hover_data=df_iris.columns,
        title="Violin plot: Sepal-length as attribute",
    )
    figure.show()

    # (iii) Petal-width:
    figure = px.violin(
        df_iris,
        x="class_type",
        y="petal_width",
        color="class_type",
        box=True,
        points="all",
        hover_data=df_iris.columns,
        title="Violin plot: Petal-width as attribute",
    )
    figure.show()

    # (iv) Petal-length:
    figure = px.violin(
        df_iris,
        x="class_type",
        y="petal_length",
        color="class_type",
        box=True,
        points="all",
        hover_data=df_iris.columns,
        title="Violin plot: Petal-length as attribute",
    )
    figure.show()

    # (3) HISTOGRAM plots:
    # (i) Sepal-width:
    figure = px.histogram(
        df_iris,
        x="sepal_width",
        color="class_type",
        title="Histogram: Sepal-width as attribute",
    )
    figure.show()

    # (ii) Sepal-length:
    figure = px.histogram(
        df_iris,
        x="sepal_length",
        color="class_type",
        title="Histogram: Sepal-length as attribute",
    )
    figure.show()

    # (iii) Petal-width:
    figure = px.histogram(
        df_iris,
        x="petal_width",
        color="class_type",
        title="Histogram: Petal-width as attribute",
    )
    figure.show()

    # (iv) Petal-length:
    figure = px.histogram(
        df_iris,
        x="petal_length",
        color="class_type",
        title="Histogram: Petal-length as attribute",
    )
    figure.show()

    # Suggested by : My code buddy - 'Luis Sosa'
    # Addition / Modification : Suggested to add 2 new plot types from plotly.

    # (4) PIE plot:
    figure = px.pie(
        df_iris,
        values="petal_length",
        names="class_type",
        color_discrete_sequence=px.colors.sequential.RdBu,
        title="Pie chart : Petal-length distribution by Class",
    )
    figure.show()

    # (5) BOX plots:
    # (i) Sepal-width:
    figure = px.box(
        df_iris,
        x="class_type",
        y="sepal_width",
        color="class_type",
        points="all",
        hover_data=df_iris.columns,
        title="Box plot: Sepal-width as attribute",
    )
    figure.show()

    # (ii) Sepal-length:
    figure = px.box(
        df_iris,
        x="class_type",
        y="sepal_length",
        color="class_type",
        points="all",
        hover_data=df_iris.columns,
        title="Box plot: Sepal-length as attribute",
    )
    figure.show()

    # (iii) Petal-width:
    figure = px.box(
        df_iris,
        x="class_type",
        y="petal_width",
        color="class_type",
        points="all",
        hover_data=df_iris.columns,
        title="Box plot: Petal-width as attribute",
    )
    figure.show()

    # (iv) Petal-length:
    figure = px.box(
        df_iris,
        x="class_type",
        y="petal_length",
        color="class_type",
        points="all",
        hover_data=df_iris.columns,
        title="Box: Petal-length as attribute",
    )
    figure.show()

    # -----------------------------------------------------------------------------------------------------

    # (D) Building models on the data

    # I have used 3 models for analyzing:
    # (1) Random Forest Classifier
    # (2) Logistic Regression
    # (3) Linear Discriminant Analysis (LDA) Classifier

    # Dividing the data into Training-set and Testing-set
    X_train, X_test, y_train, y_test = train_test_split(
        df_iris.iloc[:, :-1].values,
        df_iris["class_type"],
        test_size=0.3,
        random_state=100,
    )

    # Training-set and Testing-set: data size
    print(
        "\n\n\n ########## Size of Training data-set and Testing data-set ##########\n"
    )
    print("Training data-set size: ", X_train.shape)
    print("Testing data-set size: ", X_test.shape)

    # Suggested by : My code buddy - 'Luis Sosa'
    # Addition / Modification : Use "StandardScaler()" instead of "Normalizer()".

    # (i) Random Forest Classifier:

    # Initializing the pipeline for normalizing data using RandomForest Classifier
    pipeline_random_forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=1234)),
        ]
    )

    # Implementing pipeline to normalize the data using RandomForest Classifier
    pipeline_random_forest.fit(X_train, y_train)

    # Prediction for the Testing-data using RandomForest Classifier
    predict_random_forest = pipeline_random_forest.predict(X_test)

    # Displaying model accuracy results
    print("\n\n\n########## Model Accuracy Results ##########\n")

    # Displaying the Accuracy results of RandomForest Classifier model
    print("(1) Model accuracy using Random Forest Classifier:")
    accuracy_random_forest = accuracy_score(y_test, predict_random_forest)
    print(f"Accuracy is: {accuracy_random_forest}")

    # (ii) Logistic Regression:

    # Initializing the pipeline for normalizing data using Logistic Regression
    pipeline_logistic_regression = Pipeline(
        [("scaler", StandardScaler()), ("lf_fit", LogisticRegression())]
    )

    # Implementing pipeline to normalize the data using LogisticRegression
    pipeline_logistic_regression.fit(X_train, y_train)

    # Prediction for the Testing-data using LogisticRegression
    predict_logistic_regression = pipeline_logistic_regression.predict(X_test)

    # Displaying the Accuracy results of LogisticRegression model
    print("\n(2) Model accuracy using Logistic Regression:")
    accuracy_logistic_regression = accuracy_score(y_test, predict_logistic_regression)
    print(f"Accuracy is: {accuracy_logistic_regression}")

    # (iii) Linear Discriminant Analysis (LDA) Classifier:

    # Initializing the pipeline for normalizing data using Linear Discriminant Analysis Classifier
    pipeline_lda_classifier = Pipeline(
        [("scaler", StandardScaler()), ("lda", LDA(n_components=1))]
    )

    # Implementing pipeline to normalize the data using LDA Classifier
    pipeline_lda_classifier.fit(X_train, y_train)

    # Prediction for the Testing-data using LDA Classifier
    predict_lda_classifier = pipeline_lda_classifier.predict(X_test)

    # Displaying the Accuracy results of LDA Classifier model
    print("\n(3) Model accuracy using Linear Discriminant Analysis (LDA) Classifier:")
    accuracy_lda_classifier = accuracy_score(y_test, predict_lda_classifier)
    print(f"Accuracy is: {accuracy_lda_classifier}\n")


if __name__ == "__main__":
    sys.exit(main())
