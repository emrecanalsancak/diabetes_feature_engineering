import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("dataset/diabetes.csv")
df.head()
df.info()


# Checking the dataframe
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# Numerical and categorical variables in the dataset.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optional
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat inside cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]  # 0,1,2
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]  # name
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Analysis of categorical and numerical variables.
def cat_summary(dataframe, col_name, plot=False):
    """
    This function generates a summary of a categorical variable in a pandas DataFrame.

    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the categorical variable
    - col_name (str): the name of the categorical variable to be summarized
    - plot (bool, optional): if True, a countplot will be generated to visualize the distribution of the categorical variable. Defaults to False.

    Returns:
    - None. The function only prints the summary table and, optionally, the countplot.

    Example Usage:
    cat_summary(df, 'color', plot=True)

    This will generate a summary table and a countplot of the 'color' variable in the 'df' DataFrame.
    """
    print(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Outcome")


def num_summary(dataframe, numerical_col, plot=False):
    """
    This function generates a summary of a numerical variable in a pandas DataFrame.

    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the numerical variable
    - numerical_col (str): the name of the numerical variable to be summarized
    - plot (bool, optional): if True, a histogram will be generated to visualize the distribution of the numerical variable. Defaults to False.

    Returns:
    - None. The function only prints the summary table and, optionally, the histogram.

    Example Usage:
    num_summary(df, 'age', plot=True)

    This will generate a summary table and a histogram of the 'age' variable in the 'df' DataFrame.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=False)


# Summary of numerical variables with target variable
def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function generates a summary of a numerical variable grouped by a binary target variable in a pandas DataFrame.

    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the target and numerical variables
    - target (str): the name of the binary target variable to group by
    - numerical_col (str): the name of the numerical variable to be summarized

    Returns:
    - None. The function only prints the mean of the numerical variable for each target group.

    Example Usage:
    target_summary_with_num(df, 'is_customer', 'spending')

    This will generate the mean spending for each customer group in the 'df' DataFrame.
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Handling missing values

# Note: Variables that could not be zero were assigned zero.
# It is known that a human cannot have variable values 0 other than Pregnancies and Outcome.
# Therefore, an action decision should be taken regarding these values.
# Values that are 0 can be assigned NaN.

df.isnull().sum()

zero_columns = [
    col
    for col in df.columns
    if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])
]

zero_columns

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    """
    This function generates a summary table of missing values in a pandas DataFrame.

    Args:
    - dataframe (pandas DataFrame): the DataFrame to check for missing values
    - na_name (bool, optional): if True, returns the list of columns with missing values. Defaults to False.

    Returns:
    - missing_df (pandas DataFrame): a summary table of missing values, including the number of missing values and the ratio of missing values to the total number of rows in the DataFrame.
    - na_columns (list, optional): a list of columns with missing values. Only returned if na_name=True.

    Example Usage:
    missing_values_table(df, na_name=True)

    This will generate a summary table of missing values in the 'df' DataFrame and return the list of columns with missing values.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (
        dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100
    ).sort_values(ascending=False)
    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]
    )
    if na_name:
        return missing_df, na_columns
    return missing_df


misisng_df, na_columns = missing_values_table(df, na_name=True)


def missing_vs_target(dataframe, target, na_columns):
    """
    This function generates summary statistics of the relationship between missing values and a target variable in a pandas DataFrame.

    Args:
    - dataframe (pandas DataFrame): the DataFrame containing the target variable and columns with missing values
    - target (str): the name of the target variable in the DataFrame
    - na_columns (list): a list of column names with missing values in the DataFrame

    Returns:
    - None. The function only generates and displays summary statistics for each column with missing values.

    Example Usage:
    missing_vs_target(df, 'target', ['column1', 'column2'])

    This will generate and display summary statistics for the relationship between missing values in 'column1' and 'column2' and the 'target' variable in the 'df' DataFrame.
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(
            pd.DataFrame(
                {
                    "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                    "Count": temp_df.groupby(col)[target].count(),
                }
            ),
            end="\n\n\n",
        )


missing_vs_target(df, "Outcome", na_columns)


def filling_missin_values_KNN(df, cat_cols, num_cols, zero_columns):
    """
    This function prepares a pandas DataFrame for machine learning by performing one-hot encoding on categorical variables, scaling numerical variables, and imputing missing values using K-Nearest Neighbors (KNN) imputation.

    Args:
    - df (pandas DataFrame): the DataFrame to prepare for machine learning
    - cat_cols (list): a list of column names containing categorical variables
    - num_cols (list): a list of column names containing numerical variables
    - zero_columns (list): a list of column names containing variables that should not be imputed if they have missing values

    Returns:
    - df (pandas DataFrame): a machine learning-ready DataFrame with one-hot encoded categorical variables, scaled numerical variables, and imputed missing values.

    Example Usage:
    prepare_data_for_ml(df, ['cat_var1', 'cat_var2'], ['num_var1', 'num_var2'], ['target'])

    This will prepare the 'df' DataFrame for machine learning by performing one-hot encoding on 'cat_var1' and 'cat_var2', scaling 'num_var1' and 'num_var2', and imputing missing values using KNN imputation, except for the 'target' variable.
    """
    # One-hot encode categorical variables
    dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

    # Scale numerical variables
    scaler = MinMaxScaler()
    dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

    # Impute missing values with KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

    # Scale back to original range
    dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

    # Replace missing values in original dataframe with imputed values
    for col in zero_columns:
        df.loc[df[col].isnull(), col] = dff[[col]]

    return df


df = filling_missin_values_KNN(df, cat_cols, num_cols, zero_columns)
df.isnull().sum()
df.describe()


# Outliers
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates the lower and upper limits for outliers of a numerical column in a given dataframe using the interquartile range (IQR) method.

    Parameters:
    -----------
    - dataframe: pandas DataFrame
        The dataframe containing the column to be checked for outliers.
    - col_name: str
        The name of the numerical column to be checked for outliers.
    - q1: float, optional (default=0.05)
        The lower percentile value to calculate the first quartile.
    - q3: float, optional (default=0.95)
        The upper percentile value to calculate the third quartile.

    Returns:
    --------
    Tuple containing the lower and upper limits for outliers.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Check if a given column in a given dataframe contains any outliers.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to check for outliers.
    col_name (str): The name of the column to check for outliers.

    Returns:
    bool: True if the column contains at least one outlier, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    Replace the values of a numerical variable in a given dataframe with the corresponding upper and lower limits
    if the value is above or below the outlier thresholds respectively.

    Parameters:
    dataframe: pandas.DataFrame
    The dataframe which contains the variable to be replaced.
    variable: str
    The name of the numerical variable to be replaced.
    q1: float, optional (default=0.05)
    The percentile value representing the lower limit of the outlier thresholds.
    q3: float, optional (default=0.95)
    The percentile value representing the upper limit of the outlier thresholds.

    Returns:
    None
    The function replaces the values in place without returning anything.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

df.isnull().sum()
df.describe()


# Correlation analysis

df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# Creating new variables
df.head()

# Body Mass Index (BMI) Categories:
# Underweight - Normal - Overweight - Obese

df.loc[(df["Age"] >= 21) & df["Age"] < 50, "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"
df.sort_values(by="Age", ascending=True)

df["Insulin"].isnull().sum()
df["NEW_BMI"] = pd.cut(
    x=df["BMI"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=["Underweight", "Healthy", "Overweight", "Obese"],
)

df["NEW_GLUCOSE"] = pd.cut(
    x=df["Glucose"],
    bins=[0, 140, 200, 300],
    labels=["Normal", "Prediabetes", "Diabetes"],
)

df.loc[
    (df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"
] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[
    ((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)),
    "NEW_AGE_BMI_NOM",
] = "healthymature"
df.loc[
    ((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"
] = "healthysenior"
df.loc[
    ((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)),
    "NEW_AGE_BMI_NOM",
] = "overweightmature"
df.loc[
    ((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"
] = "overweightsenior"
df.loc[
    (df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"
] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

df.loc[
    (df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"
] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[
    ((df["Glucose"] >= 70) & (df["Glucose"] < 100))
    & ((df["Age"] >= 21) & (df["Age"] < 50)),
    "NEW_AGE_GLUCOSE_NOM",
] = "normalmature"
df.loc[
    ((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50),
    "NEW_AGE_GLUCOSE_NOM",
] = "normalsenior"
df.loc[
    ((df["Glucose"] >= 100) & (df["Glucose"] <= 125))
    & ((df["Age"] >= 21) & (df["Age"] < 50)),
    "NEW_AGE_GLUCOSE_NOM",
] = "hiddenmature"
df.loc[
    ((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50),
    "NEW_AGE_GLUCOSE_NOM",
] = "hiddensenior"
df.loc[
    (df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)),
    "NEW_AGE_GLUCOSE_NOM",
] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"
df["blood_pressure_category"] = np.where(
    (df["BloodPressure"] < 80) | (df["BloodPressure"] >= 120), "abnormal", "normal"
)


def set_insulin(dataframe, col_name="Insulin"):
    """
    This function takes a pandas DataFrame and a column name containing insulin levels, and returns a string indicating whether the insulin level is "Normal" or "Abnormal".

    Insulin levels between 16 and 166 are considered normal, while levels outside this range are considered abnormal.

    Parameters:

    dataframe (pandas DataFrame): the DataFrame containing the insulin level column to be analyzed
    col_name (str): the name of the column containing insulin level values (default is "Insulin")
    Returns:

    insulin_status (str): a string indicating whether the insulin level is "Normal" or "Abnormal"
    """
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df.columns
df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df["Avg_Glucose_Per_Age"] = df["Glucose"] / df["Age"]

df["High_BP"] = ((df["BloodPressure"] > 140)).astype(int)

df.columns = [col.upper() for col in df.columns]

# Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# LABEL ENCODING


def label_encoder(dataframe, binary_col):
    """
    This function takes a pandas DataFrame and the name of a binary column containing categorical variables, and performs label encoding on the specified column. Label encoding converts each unique value in a categorical variable to a numerical label.

    Parameters:

    dataframe (pandas DataFrame): the DataFrame to encode
    binary_col (str): the name of the binary column containing categorical variables to be label encoded
    Returns:

    encoded_dataframe (pandas DataFrame): the original DataFrame with the specified binary column replaced with label encoded numerical values
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [
    col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2
]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols = [
    col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]
]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    This function takes a pandas DataFrame and a list of column names containing categorical variables, and performs one-hot encoding on the specified columns. One-hot encoding creates binary columns for each unique value in a categorical variable, replacing the original column with the new binary columns.

    Parameters:

    dataframe (pandas DataFrame): the DataFrame to encode
    categorical_cols (list): a list of column names containing categorical variables to be one-hot encoded
    drop_first (bool): whether to drop the first binary column for each categorical variable (default is False)
    Returns:

    encoded_dataframe (pandas DataFrame): the original DataFrame with the specified categorical columns replaced with one-hot encoded binary columns
    """
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

##################################
# STANDARDIZATION
##################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape
df.describe()


df = pd.read_csv("dataset/diabetes.csv")
df.head()

# MODELLING
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


# Before feature engineering the accuracy was: 0.75
# After feature engineering accuracy: 0.78
