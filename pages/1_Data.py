import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the current working directory
current_dir = os.getcwd()

# Relative path to the dataset within the "dataset" folder
dataset_path = os.path.join(current_dir, 'dataset', 'Churn Prediction Dataset.csv')

# Load the dataset
df = pd.read_csv(dataset_path)

# Create a display table showing basic information about the dataset
styled_info_table = df.style.set_table_styles([
    {'selector': 'th', 'props': [('color', 'white'), ('background-color', '#3498db')]},  # Blue headers
    {'selector': 'td', 'props': [('background-color', '#ecf0f1')]}  # Light grey cells
]).set_caption("Basic Information about the Dataset")
styled_info_table

# Create a display table showing summary statistics of numerical variables
styled_summary_table = df.describe().style.set_table_styles([
    {'selector': 'th', 'props': [('color', 'white'), ('background-color', '#27ae60')]},  # Green headers
    {'selector': 'td', 'props': [('background-color', '#f1c40f')]}  # Yellow cells
]).set_caption("Summary Statistics of Numerical Variables")
styled_summary_table

# Create a display table showing the first few rows of the dataset
styled_head_table = df.head().style.set_table_styles([
    {'selector': 'th', 'props': [('color', 'white'), ('background-color', '#e74c3c')]},  # Red headers
    {'selector': 'td', 'props': [('background-color', '#95a5a6')]}  # Grey cells
]).set_caption("First Few Rows of the Dataset")
styled_head_table

# Check for missing values and create a display table
missing_values_df = pd.DataFrame({'Column': df.columns, 'Missing Values': df.isnull().sum()})
styled_missing_table = missing_values_df.style.set_table_styles([
    {'selector': 'th', 'props': [('color', 'white'), ('background-color', '#8e44ad')]},  # Purple headers
    {'selector': 'td', 'props': [('background-color', '#d7bde2')]}  # Lavender cells
]).set_caption("Missing Values")
styled_missing_table

# Check for duplicate rows and create a display table
duplicate_count = df.duplicated().sum()
styled_duplicate_table = pd.DataFrame({'Duplicate Rows': [duplicate_count]}).style.set_table_styles([
    {'selector': 'th', 'props': [('color', 'white'), ('background-color', '#f39c12')]},  # Orange headers
    {'selector': 'td', 'props': [('background-color', '#f9e79f')]}  # Light yellow cells
]).set_caption("Duplicate Rows")
styled_duplicate_table

# Univariate Analysis

# Visualize the distribution of the target variable using Plotly Express
fig_churn_distribution = px.histogram(df, x='Churn', title='Distribution of Churn')
fig_churn_distribution.update_traces(marker_color='#3498db')  # Blue color for bars
fig_churn_distribution.update_layout(title_font_size=20)
fig_churn_distribution.show()

# Bivariate Analysis

# Visualize the relationship between 'Churn' and 'MonthlyCharges' using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('Monthly Charges vs Churn', fontsize=20)
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Monthly Charges', fontsize=14)
plt.show()

# Additional Analysis using pandas.styling

# Create a styled table based on the entire dataset
styled_filtered_table = (
    df.style
    .format({'MonthlyCharges': '${:,.2f}', 'TotalCharges': '${:,.2f}'})  # Currency formatting
    .format({'Tenure': '{:,d}'})  # Integer formatting
    .format({'ChurnRate': '{:.2%}'})  # Percentage formatting
    .format({'CustomerID': '{:,.0f}'})  # Compact number formatting
    .hide(subset=['Gender', 'SeniorCitizen'])  # Hide certain columns
    .set_table_styles([
        {'selector': 'th.row_heading', 'props': [('background-color', '#3498db'), ('color', 'white')]},  # Blue row headers
        {'selector': 'th.col_heading', 'props': [('background-color', '#e74c3c'), ('color', 'white')]},  # Red column headers
        {'selector': 'td', 'props': [('background-color', '#ecf0f1')]}  # Light grey cells
    ])
    .set_caption("Churn Prediction Dataset")
    .set_properties(**{
        'text-align': 'center',  # Center align all cells
        'border': '1px solid black'  # Add a black border
    })
)
styled_filtered_table