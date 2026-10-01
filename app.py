# ============================================================
# BAD702 - HR EMPLOYEE ATTRITION ANALYTICS DASHBOARD
# Complete Streamlit Application
# Modules 1 to 5 + Final Machine Learning Comparison
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BAD702 HR Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 BAD702 HR Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Statistical and Machine Learning Analysis of Employee Attrition'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 Project Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "📊 Module 1 - EDA",
        "📈 Module 2 - Sampling",
        "🧪 Module 3 - Hypothesis Testing",
        "📉 Module 4 - Regression",
        "🤖 Module 5 - LDA Classification",
        "🏆 Final ML Comparison"
    ]
)


# ============================================================
# DATA LOADING
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload HR Employee Attrition CSV",
    type=["csv"]
)

# Try uploaded file first
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    # Try default file
    try:
        df = pd.read_csv(
            "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        )

    except FileNotFoundError:

        st.warning(
            "Please upload the HR Employee Attrition CSV "
            "from the sidebar."
        )

        st.stop()


# ============================================================
# BASIC DATA PREPARATION
# ============================================================

# Remove constant columns and identifier
columns_to_remove = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber"
]

existing_remove = [
    col for col in columns_to_remove
    if col in df.columns
]

df_clean = df.drop(
    columns=existing_remove
)


# ============================================================
# COMMON VARIABLES
# ============================================================

numeric_columns = df_clean.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df_clean.select_dtypes(
    include="object"
).columns.tolist()


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.header("🏠 Project Overview")

    st.markdown("""
    ### Project Title

    **Statistical and Machine Learning Analysis of Employee Attrition Using HR Data**

    ### Main Research Question

    **What factors are associated with employee attrition,
    and how accurately can employee attrition be predicted
    using statistical and machine-learning techniques?**
    """)

    st.markdown("---")

    # Dataset metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Employees",
            len(df)
        )

    with col2:
        st.metric(
            "Original Columns",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📚 Project Modules")

    module_info = pd.DataFrame({
        "Module": [
            "Module 1",
            "Module 2",
            "Module 3",
            "Module 4",
            "Module 5",
            "Final Analysis"
        ],

        "Topic": [
            "Data Cleaning & Exploratory Data Analysis",
            "Sampling & Sampling Distributions",
            "Hypothesis Testing & A/B Testing",
            "Regression Analysis",
            "Discriminant Analysis & Classification",
            "Machine Learning Model Comparison"
        ]
    })

    st.dataframe(
        module_info,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "All five statistical modules and the final ML "
        "analysis are available from the sidebar."
    )


# ============================================================
# MODULE 1 - EDA
# ============================================================

elif page == "📊 Module 1 - EDA":

    st.header("📊 Module 1 - Data Cleaning & Exploratory Data Analysis")

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.subheader("1. Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Original Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Cleaned Columns",
            df_clean.shape[1]
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    st.subheader("2. Missing Value Analysis")

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().sum().values /
            len(df) * 100
        )
    })

    st.dataframe(
        missing,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Duplicate values
    # --------------------------------------------------------

    st.subheader("3. Duplicate Analysis")

    duplicate_count = df.duplicated().sum()

    if duplicate_count == 0:
        st.success("No duplicate rows found.")

    else:
        st.warning(
            f"{duplicate_count} duplicate rows found."
        )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    st.subheader("4. Descriptive Statistics")

    st.dataframe(
        df_clean.describe().T,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Attrition distribution
    # --------------------------------------------------------

    st.subheader("5. Employee Attrition Distribution")

    attrition_counts = df_clean[
        "Attrition"
    ].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            attrition_counts.to_frame(
                "Employee Count"
            )
        )

    with col2:

        fig, ax = plt.subplots()

        ax.pie(
            attrition_counts.values,
            labels=attrition_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Employee Attrition Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Categorical analysis
    # --------------------------------------------------------

    st.subheader("6. Categorical Analysis")

    selected_category = st.selectbox(
        "Select categorical variable",
        [
            "BusinessTravel",
            "Department",
            "EducationField",
            "Gender",
            "JobRole",
            "MaritalStatus",
            "OverTime"
        ]
    )

    category_counts = df_clean[
        selected_category
    ].value_counts()

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=ax
    )

    ax.set_title(
        f"{selected_category} Distribution"
    )

    ax.set_xlabel(
        selected_category
    )

    ax.set_ylabel(
        "Number of Employees"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Attrition by category
    # --------------------------------------------------------

    st.subheader(
        "7. Attrition by Categorical Variable"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.countplot(
        data=df_clean,
        x=selected_category,
        hue="Attrition",
        ax=ax
    )

    ax.set_title(
        f"Attrition by {selected_category}"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    st.subheader("8. Histogram Analysis")

    selected_numeric = st.selectbox(
        "Select numerical variable",
        numeric_columns
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        data=df_clean,
        x=selected_numeric,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_numeric}"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------

    st.subheader("9. Boxplot Analysis")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=df_clean,
        y=selected_numeric,
        ax=ax
    )

    ax.set_title(
        f"Boxplot of {selected_numeric}"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Violin plot
    # --------------------------------------------------------

    st.subheader("10. Violin Plot")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.violinplot(
        data=df_clean,
        x="Attrition",
        y=selected_numeric,
        ax=ax
    )

    ax.set_title(
        f"{selected_numeric} by Attrition"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    st.subheader("11. Scatter Plot")

    scatter_x = st.selectbox(
        "X-axis variable",
        numeric_columns,
        index=0
    )

    scatter_y = st.selectbox(
        "Y-axis variable",
        numeric_columns,
        index=min(
            1,
            len(numeric_columns) - 1
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=df_clean,
        x=scatter_x,
        y=scatter_y,
        hue="Attrition",
        ax=ax
    )

    ax.set_title(
        f"{scatter_x} vs {scatter_y}"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    st.subheader("12. Correlation Heatmap")

    correlation = df_clean[
        numeric_columns
    ].corr()

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Outlier analysis
    # --------------------------------------------------------

    st.subheader("13. Outlier Analysis using IQR")

    outlier_summary = []

    for column in numeric_columns:

        Q1 = df_clean[column].quantile(
            0.25
        )

        Q3 = df_clean[column].quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = (
            (df_clean[column] < lower) |
            (df_clean[column] > upper)
        ).sum()

        outlier_summary.append({
            "Variable": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Number of Outliers": count
        })

    outlier_table = pd.DataFrame(
        outlier_summary
    )

    st.dataframe(
        outlier_table.round(2),
        use_container_width=True
    )


# ============================================================
# MODULE 2 - SAMPLING
# ============================================================

elif page == "📈 Module 2 - Sampling":

    st.header(
        "📈 Module 2 - Data & Sampling Distributions"
    )

    population = df_clean[
        "MonthlyIncome"
    ].dropna()

    sample_size = st.slider(
        "Select sample size",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )

    number_of_samples = st.slider(
        "Number of samples",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    st.subheader("1. Population")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Population Size",
            len(population)
        )

    with col2:
        st.metric(
            "Population Mean",
            f"{population.mean():.2f}"
        )

    with col3:
        st.metric(
            "Population Std. Dev.",
            f"{population.std():.2f}"
        )

    # --------------------------------------------------------
    # Random sample
    # --------------------------------------------------------

    sample = population.sample(
        n=sample_size,
        random_state=42
    )

    sample_mean = sample.mean()
    sample_std = sample.std(
        ddof=1
    )

    st.subheader("2. Random Sample")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sample Size",
            sample_size
        )

    with col2:
        st.metric(
            "Sample Mean",
            f"{sample_mean:.2f}"
        )

    with col3:
        st.metric(
            "Sample Std. Dev.",
            f"{sample_std:.2f}"
        )

    # --------------------------------------------------------
    # Sampling distribution
    # --------------------------------------------------------

    sample_means = []

    for i in range(
        number_of_samples
    ):

        random_sample = population.sample(
            n=sample_size,
            replace=True,
            random_state=i
        )

        sample_means.append(
            random_sample.mean()
        )

    sampling_distribution = pd.Series(
        sample_means
    )

    st.subheader(
        "3. Sampling Distribution of Sample Means"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sampling Distribution Mean",
            f"{sampling_distribution.mean():.2f}"
        )

    with col2:
        st.metric(
            "Observed Standard Error",
            f"{sampling_distribution.std():.2f}"
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        sampling_distribution,
        bins=30,
        kde=True,
        ax=ax
    )

    ax.axvline(
        population.mean(),
        linestyle="--",
        linewidth=2,
        label="Population Mean"
    )

    ax.axvline(
        sampling_distribution.mean(),
        linestyle="--",
        linewidth=2,
        label="Sampling Mean"
    )

    ax.set_title(
        "Sampling Distribution of Monthly Income"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Standard error
    # --------------------------------------------------------

    theoretical_se = (
        population.std(ddof=1) /
        np.sqrt(sample_size)
    )

    st.subheader(
        "4. Standard Error"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Theoretical SE",
            f"{theoretical_se:.2f}"
        )

    with col2:
        st.metric(
            "Observed SE",
            f"{sampling_distribution.std():.2f}"
        )

    # --------------------------------------------------------
    # t-distribution
    # --------------------------------------------------------

    degrees_freedom = (
        sample_size - 1
    )

    confidence_level = 0.95

    t_critical = stats.t.ppf(
        1 - 0.05 / 2,
        degrees_freedom
    )

    margin_error = (
        t_critical *
        sample_std /
        np.sqrt(sample_size)
    )

    lower_ci = (
        sample_mean -
        margin_error
    )

    upper_ci = (
        sample_mean +
        margin_error
    )

    st.subheader(
        "5. Student's t-Distribution & 95% Confidence Interval"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Degrees of Freedom",
            degrees_freedom
        )

    with col2:
        st.metric(
            "t-Critical",
            f"{t_critical:.4f}"
        )

    with col3:
        st.metric(
            "Lower CI",
            f"{lower_ci:.2f}"
        )

    with col4:
        st.metric(
            "Upper CI",
            f"{upper_ci:.2f}"
        )

    st.info(
        f"95% Confidence Interval for mean Monthly Income: "
        f"{lower_ci:.2f} to {upper_ci:.2f}"
    )


# ============================================================
# MODULE 3 - HYPOTHESIS TESTING
# ============================================================

elif page == "🧪 Module 3 - Hypothesis Testing":

    st.header(
        "🧪 Module 3 - Statistical Experiments & Significance Testing"
    )

    alpha = 0.05

    st.info(
        "Significance level α = 0.05"
    )

    # --------------------------------------------------------
    # One sample t-test
    # --------------------------------------------------------

    st.subheader(
        "1. One-Sample T-Test"
    )

    monthly_income = df_clean[
        "MonthlyIncome"
    ].dropna()

    reference_mean = st.number_input(
        "Reference Monthly Income",
        min_value=0,
        value=5000,
        step=500
    )

    t_one, p_one = stats.ttest_1samp(
        monthly_income,
        reference_mean
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "t-statistic",
            f"{t_one:.4f}"
        )

    with col2:
        st.metric(
            "p-value",
            f"{p_one:.6f}"
        )

    if p_one < alpha:
        st.error(
            "Decision: Reject H₀. "
            "The mean Monthly Income differs significantly "
            "from the reference value."
        )
    else:
        st.success(
            "Decision: Fail to Reject H₀. "
            "There is not enough evidence that the mean "
            "differs from the reference value."
        )

    # --------------------------------------------------------
    # Independent t-test
    # --------------------------------------------------------

    st.subheader(
        "2. Independent Two-Sample T-Test"
    )

    income_no = df_clean[
        df_clean["Attrition"] == "No"
    ]["MonthlyIncome"]

    income_yes = df_clean[
        df_clean["Attrition"] == "Yes"
    ]["MonthlyIncome"]

    t_two, p_two = stats.ttest_ind(
        income_no,
        income_yes,
        equal_var=False
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean - Stayed",
            f"{income_no.mean():.2f}"
        )

    with col2:
        st.metric(
            "Mean - Left",
            f"{income_yes.mean():.2f}"
        )

    with col3:
        st.metric(
            "p-value",
            f"{p_two:.6f}"
        )

    if p_two < alpha:
        st.warning(
            "Statistically significant difference detected "
            "between the two groups."
        )
    else:
        st.info(
            "No statistically significant difference detected."
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.boxplot(
        data=df_clean,
        x="Attrition",
        y="MonthlyIncome",
        ax=ax
    )

    ax.set_title(
        "Monthly Income by Attrition"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Overtime t-test
    # --------------------------------------------------------

    st.subheader(
        "3. Overtime vs Monthly Income T-Test"
    )

    overtime_yes = df_clean[
        df_clean["OverTime"] == "Yes"
    ]["MonthlyIncome"]

    overtime_no = df_clean[
        df_clean["OverTime"] == "No"
    ]["MonthlyIncome"]

    t_ot, p_ot = stats.ttest_ind(
        overtime_yes,
        overtime_no,
        equal_var=False
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Overtime = Yes",
            f"{overtime_yes.mean():.2f}"
        )

    with col2:
        st.metric(
            "Overtime = No",
            f"{overtime_no.mean():.2f}"
        )

    with col3:
        st.metric(
            "p-value",
            f"{p_ot:.6f}"
        )

    # --------------------------------------------------------
    # Chi-square: overtime
    # --------------------------------------------------------

    st.subheader(
        "4. Chi-Square Test: OverTime vs Attrition"
    )

    overtime_table = pd.crosstab(
        df_clean["OverTime"],
        df_clean["Attrition"]
    )

    chi_ot, p_chi_ot, dof_ot, expected_ot = (
        stats.chi2_contingency(
            overtime_table
        )
    )

    st.dataframe(
        overtime_table,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Chi-Square",
            f"{chi_ot:.4f}"
        )

    with col2:
        st.metric(
            "Degrees of Freedom",
            dof_ot
        )

    with col3:
        st.metric(
            "p-value",
            f"{p_chi_ot:.6f}"
        )

    if p_chi_ot < alpha:
        st.warning(
            "There is a statistically significant association "
            "between OverTime and Attrition."
        )
    else:
        st.info(
            "No statistically significant association was detected."
        )

    # --------------------------------------------------------
    # Chi-square: Job Role
    # --------------------------------------------------------

    st.subheader(
        "5. Chi-Square Test: JobRole vs Attrition"
    )

    job_table = pd.crosstab(
        df_clean["JobRole"],
        df_clean["Attrition"]
    )

    chi_job, p_job, dof_job, expected_job = (
        stats.chi2_contingency(
            job_table
        )
    )

    st.dataframe(
        job_table,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Chi-Square",
            f"{chi_job:.4f}"
        )

    with col2:
        st.metric(
            "Degrees of Freedom",
            dof_job
        )

    with col3:
        st.metric(
            "p-value",
            f"{p_job:.6f}"
        )

    # --------------------------------------------------------
    # A/B testing
    # --------------------------------------------------------

    st.subheader(
        "6. A/B Testing: Overtime and Attrition"
    )

    group_a = df_clean[
        df_clean["OverTime"] == "No"
    ]

    group_b = df_clean[
        df_clean["OverTime"] == "Yes"
    ]

    attrition_a = (
        group_a["Attrition"] == "Yes"
    ).sum()

    attrition_b = (
        group_b["Attrition"] == "Yes"
    ).sum()

    total_a = len(group_a)
    total_b = len(group_b)

    rate_a = attrition_a / total_a
    rate_b = attrition_b / total_b

    pooled = (
        (attrition_a + attrition_b) /
        (total_a + total_b)
    )

    standard_error = np.sqrt(
        pooled *
        (1 - pooled) *
        (
            1 / total_a +
            1 / total_b
        )
    )

    z_stat = (
        rate_a - rate_b
    ) / standard_error

    p_ab = (
        2 *
        (1 - stats.norm.cdf(abs(z_stat)))
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "No Overtime Attrition Rate",
            f"{rate_a * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Overtime Attrition Rate",
            f"{rate_b * 100:.2f}%"
        )

    with col3:
        st.metric(
            "p-value",
            f"{p_ab:.6f}"
        )

    if p_ab < alpha:
        st.warning(
            "The difference in attrition rates is statistically significant."
        )
    else:
        st.info(
            "The difference in attrition rates is not statistically significant."
        )

    ab_data = pd.DataFrame({
        "Group": [
            "No Overtime",
            "Overtime"
        ],
        "Attrition Rate": [
            rate_a * 100,
            rate_b * 100
        ]
    })

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        data=ab_data,
        x="Group",
        y="Attrition Rate",
        ax=ax
    )

    ax.set_title(
        "A/B Test: Attrition Rate"
    )

    ax.set_ylabel(
        "Attrition Rate (%)"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MODULE 4 - REGRESSION
# ============================================================

elif page == "📉 Module 4 - Regression":

    st.header(
        "📉 Module 4 - Regression Analysis"
    )

    # --------------------------------------------------------
    # Simple Regression
    # --------------------------------------------------------

    st.subheader(
        "1. Simple Linear Regression"
    )

    X_simple = df_clean[
        ["TotalWorkingYears"]
    ]

    y_simple = df_clean[
        "MonthlyIncome"
    ]

    X_train_s, X_test_s, y_train_s, y_test_s = (
        train_test_split(
            X_simple,
            y_simple,
            test_size=0.20,
            random_state=42
        )
    )

    simple_model = LinearRegression()

    simple_model.fit(
        X_train_s,
        y_train_s
    )

    y_pred_s = simple_model.predict(
        X_test_s
    )

    r2_s = r2_score(
        y_test_s,
        y_pred_s
    )

    mae_s = mean_absolute_error(
        y_test_s,
        y_pred_s
    )

    rmse_s = np.sqrt(
        mean_squared_error(
            y_test_s,
            y_pred_s
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R²",
            f"{r2_s:.4f}"
        )

    with col2:
        st.metric(
            "MAE",
            f"{mae_s:.2f}"
        )

    with col3:
        st.metric(
            "RMSE",
            f"{rmse_s:.2f}"
        )

    st.write(
        f"**Regression Equation:** "
        f"MonthlyIncome = "
        f"{simple_model.intercept_:.2f} + "
        f"{simple_model.coef_[0]:.2f} × TotalWorkingYears"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.scatterplot(
        x=X_test_s["TotalWorkingYears"],
        y=y_test_s,
        ax=ax
    )

    sorted_x = np.sort(
        X_test_s["TotalWorkingYears"].values
    )

    sorted_pred = simple_model.predict(
        sorted_x.reshape(-1, 1)
    )

    ax.plot(
        sorted_x,
        sorted_pred,
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Total Working Years vs Monthly Income"
    )

    ax.set_xlabel(
        "Total Working Years"
    )

    ax.set_ylabel(
        "Monthly Income"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Multiple Regression
    # --------------------------------------------------------

    st.subheader(
        "2. Multiple Linear Regression"
    )

    regression_features = [
        "Age",
        "TotalWorkingYears",
        "YearsAtCompany",
        "JobLevel",
        "JobSatisfaction",
        "DistanceFromHome",
        "NumCompaniesWorked",
        "PercentSalaryHike"
    ]

    X = df_clean[
        regression_features
    ]

    y = df_clean[
        "MonthlyIncome"
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    multiple_model = LinearRegression()

    multiple_model.fit(
        X_train,
        y_train
    )

    y_pred = multiple_model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    n = len(y_test)
    p = X_test.shape[1]

    adjusted_r2 = (
        1 -
        ((1 - r2) *
         (n - 1) /
         (n - p - 1))
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R²",
            f"{r2:.4f}"
        )

    with col2:
        st.metric(
            "Adjusted R²",
            f"{adjusted_r2:.4f}"
        )

    with col3:
        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with col4:
        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    st.subheader(
        "3. Regression Coefficients"
    )

    coefficients = pd.DataFrame({
        "Feature": regression_features,
        "Coefficient": multiple_model.coef_
    })

    st.dataframe(
        coefficients.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Actual vs predicted
    # --------------------------------------------------------

    st.subheader(
        "4. Actual vs Predicted"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.scatterplot(
        x=y_test,
        y=y_pred,
        ax=ax
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel(
        "Actual Monthly Income"
    )

    ax.set_ylabel(
        "Predicted Monthly Income"
    )

    ax.set_title(
        "Actual vs Predicted Monthly Income"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Residual analysis
    # --------------------------------------------------------

    residuals = (
        y_test -
        y_pred
    )

    st.subheader(
        "5. Residual Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            residuals,
            kde=True,
            ax=ax
        )

        ax.axvline(
            0,
            linestyle="--"
        )

        ax.set_title(
            "Residual Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.scatterplot(
            x=y_pred,
            y=residuals,
            ax=ax
        )

        ax.axhline(
            0,
            linestyle="--"
        )

        ax.set_title(
            "Residuals vs Predicted Values"
        )

        ax.set_xlabel(
            "Predicted Monthly Income"
        )

        ax.set_ylabel(
            "Residual"
        )

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Example prediction
    # --------------------------------------------------------

    st.subheader(
        "6. Example Monthly Income Prediction"
    )

    example = pd.DataFrame({
        "Age": [30],
        "TotalWorkingYears": [8],
        "YearsAtCompany": [5],
        "JobLevel": [2],
        "JobSatisfaction": [3],
        "DistanceFromHome": [10],
        "NumCompaniesWorked": [2],
        "PercentSalaryHike": [15]
    })

    prediction = multiple_model.predict(
        example
    )[0]

    st.dataframe(
        example,
        use_container_width=True
    )

    st.success(
        f"Predicted Monthly Income: ₹{prediction:,.2f}"
    )


# ============================================================
# MODULE 5 - LDA CLASSIFICATION
# ============================================================

elif page == "🤖 Module 5 - LDA Classification":

    st.header(
        "🤖 Module 5 - Discriminant Analysis & Classification"
    )

    classification_features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    X = df_clean[
        classification_features
    ]

    y = df_clean[
        "Attrition"
    ].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    lda = LinearDiscriminantAnalysis()

    lda.fit(
        X_train_scaled,
        y_train
    )

    y_pred = lda.predict(
        X_test_scaled
    )

    y_prob = lda.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    st.subheader(
        "1. LDA Performance"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with col4:
        st.metric(
            "F1-Score",
            f"{f1:.4f}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{auc:.4f}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader(
        "2. Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Stayed",
            "Left"
        ],
        yticklabels=[
            "Stayed",
            "Left"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "LDA Confusion Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.subheader(
        "3. Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Stayed",
            "Left"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    st.subheader(
        "4. ROC Curve"
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"LDA (AUC = {auc:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "LDA ROC Curve"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # LDA coefficients
    # --------------------------------------------------------

    st.subheader(
        "5. LDA Discriminant Coefficients"
    )

    lda_coefficients = pd.DataFrame({
        "Feature": classification_features,
        "Coefficient": lda.coef_[0]
    })

    lda_coefficients[
        "Absolute Coefficient"
    ] = lda_coefficients[
        "Coefficient"
    ].abs()

    lda_coefficients = (
        lda_coefficients
        .sort_values(
            "Absolute Coefficient",
            ascending=False
        )
    )

    st.dataframe(
        lda_coefficients.round(4),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.barplot(
        data=lda_coefficients,
        x="Coefficient",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "LDA Discriminant Coefficients"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# FINAL ML COMPARISON
# ============================================================

elif page == "🏆 Final ML Comparison":

    st.header(
        "🏆 Final Machine Learning Model Comparison"
    )

    st.markdown("""
    This section compares three classification models:

    - **Linear Discriminant Analysis (LDA)**
    - **Logistic Regression**
    - **Random Forest**
    """)

    classification_features = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    X = df_clean[
        classification_features
    ]

    y = df_clean[
        "Attrition"
    ].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # --------------------------------------------------------
    # LDA
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    lda = LinearDiscriminantAnalysis()

    lda.fit(
        X_train_scaled,
        y_train
    )

    lda_pred = lda.predict(
        X_test_scaled
    )

    lda_prob = lda.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic.fit(
        X_train_scaled,
        y_train
    )

    logistic_pred = logistic.predict(
        X_test_scaled
    )

    logistic_prob = logistic.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_pred = random_forest.predict(
        X_test
    )

    rf_prob = random_forest.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics function
    # --------------------------------------------------------

    def calculate_metrics(
        actual,
        predicted,
        probability
    ):

        return [
            accuracy_score(
                actual,
                predicted
            ),

            precision_score(
                actual,
                predicted,
                zero_division=0
            ),

            recall_score(
                actual,
                predicted,
                zero_division=0
            ),

            f1_score(
                actual,
                predicted,
                zero_division=0
            ),

            roc_auc_score(
                actual,
                probability
            )
        ]

    lda_metrics = calculate_metrics(
        y_test,
        lda_pred,
        lda_prob
    )

    logistic_metrics = calculate_metrics(
        y_test,
        logistic_pred,
        logistic_prob
    )

    rf_metrics = calculate_metrics(
        y_test,
        rf_pred,
        rf_prob
    )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Model": [
            "LDA",
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            lda_metrics[0],
            logistic_metrics[0],
            rf_metrics[0]
        ],

        "Precision": [
            lda_metrics[1],
            logistic_metrics[1],
            rf_metrics[1]
        ],

        "Recall": [
            lda_metrics[2],
            logistic_metrics[2],
            rf_metrics[2]
        ],

        "F1-Score": [
            lda_metrics[3],
            logistic_metrics[3],
            rf_metrics[3]
        ],

        "ROC-AUC": [
            lda_metrics[4],
            logistic_metrics[4],
            rf_metrics[4]
        ]
    })

    st.subheader(
        "1. Model Performance Comparison"
    )

    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Metric cards
    # --------------------------------------------------------

    st.subheader(
        "2. Model Scores"
    )

    for _, row in comparison.iterrows():

        st.markdown(
            f"### {row['Model']}"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Accuracy",
                f"{row['Accuracy']:.3f}"
            )

        with col2:
            st.metric(
                "Precision",
                f"{row['Precision']:.3f}"
            )

        with col3:
            st.metric(
                "Recall",
                f"{row['Recall']:.3f}"
            )

        with col4:
            st.metric(
                "F1",
                f"{row['F1-Score']:.3f}"
            )

        with col5:
            st.metric(
                "ROC-AUC",
                f"{row['ROC-AUC']:.3f}"
            )

    # --------------------------------------------------------
    # Comparison graph
    # --------------------------------------------------------

    st.subheader(
        "3. Performance Comparison Graph"
    )

    plot_data = comparison.set_index(
        "Model"
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    plot_data.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "Machine Learning Model Comparison"
    )

    plt.xticks(
        rotation=0
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # ROC curves
    # --------------------------------------------------------

    st.subheader(
        "4. ROC Curve Comparison"
    )

    lda_fpr, lda_tpr, _ = roc_curve(
        y_test,
        lda_prob
    )

    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_prob
    )

    rf_fpr, rf_tpr, _ = roc_curve(
        y_test,
        rf_prob
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.plot(
        lda_fpr,
        lda_tpr,
        linewidth=2,
        label=f"LDA (AUC={lda_metrics[4]:.3f})"
    )

    ax.plot(
        logistic_fpr,
        logistic_tpr,
        linewidth=2,
        label=f"Logistic Regression (AUC={logistic_metrics[4]:.3f})"
    )

    ax.plot(
        rf_fpr,
        rf_tpr,
        linewidth=2,
        label=f"Random Forest (AUC={rf_metrics[4]:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve Comparison"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------------

    st.subheader(
        "5. Random Forest Feature Importance"
    )

    feature_importance = pd.DataFrame({
        "Feature": classification_features,
        "Importance":
            random_forest.feature_importances_
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    st.dataframe(
        feature_importance.round(4),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Final project summary
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📌 Overall Project Summary"
    )

    st.markdown("""
    ### Research Question

    **What factors are associated with employee attrition,
    and how accurately can employee attrition be predicted
    using statistical and machine-learning techniques?**

    ### Statistical Analysis Performed

    - Exploratory Data Analysis
    - Sampling Distribution
    - Confidence Interval
    - Student's t-Test
    - Chi-Square Test
    - A/B Testing
    - Linear Regression
    - Multiple Linear Regression
    - Linear Discriminant Analysis
    - Classification Analysis

    ### Machine Learning Models

    - Linear Discriminant Analysis
    - Logistic Regression
    - Random Forest

    ### Evaluation Metrics

    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - R²
    - MAE
    - RMSE
    """)

    st.success(
        "Complete BAD702 statistical and machine-learning "
        "analysis displayed successfully."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "BAD702 - Statistical Machine Learning for Data Science"
)

st.sidebar.caption(
    "HR Employee Attrition Analysis"
)