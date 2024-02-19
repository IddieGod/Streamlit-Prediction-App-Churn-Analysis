import gc

import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score

import xgboost as xgb
import lightgbm as lgbm

import pages.Feedback
import pages.Prediction_History
import pages.User_Profile
import pages.Churn_App
import pages.Upload_Predictor
import pages.Dashboard


st.title("Telecom Customer Churn Predictor App")

st.markdown("""
The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing 
business with an entity (e.g., Business Organization). 

You will predict the churn rate of customers in a telecom company using a stored model based on XGBoost or LightGBM.

## Instructions
1. Select the classifier (model) you want to use from the dropdown box in the sidebar
2. To check the accuracy of the classifier, click on the **`Performance on Test Dataset`** button in the sidebar
3. To predict churn rate of a single observation, click on the **`Prediction on Random Instance`** button in the sidebar
4. Or you can predict churn rate by manual input from the sidebar, scroll down and click **`Predict`** button 
5. The result will be displayed in the **[Prediction Result](#prediction-result)** section
6. If you want to upload your csv file directly, on the sidebar click on the **[Upload CSV]**
""")

df_churn = pd.read_csv("C:/Users/IddieGod/Desktop/Streamlit-Prediction-App-Churn-Analysis/dataset/Telco-Customer-Churn-dataset-cleaned.csv")
df_train = pd.read_csv('C:/Users/IddieGod/Desktop/Streamlit-Prediction-App-Churn-Analysis/dataset/Telco-Customer-Churn-dataset-Train.csv', index_col=0)
df_test = pd.read_csv('C:/Users/IddieGod/Desktop/Streamlit-Prediction-App-Churn-Analysis/dataset/Telco-Customer-Churn-dataset-Test.csv', index_col=0)

st.header('Churn Data Overview')
st.write('Data Dimension: ' + str(df_churn.shape[0]) + ' rows and ' + str(df_churn.shape[1]) + ' columns.')
st.dataframe(df_churn)


@st.cache_data()
def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(download_dataset(df_churn), unsafe_allow_html=True)

st.markdown("## Prediction Result")


st.sidebar.markdown("## Predict Customer Churn Rate")
# st.sidebar.markdown("### Select a Model")
classifier_name = st.sidebar.selectbox(
    'Select a Classifier',
    ('XGBoost', 'LightGBM')
)


def get_classifier(clf_name):
    if clf_name == 'XGBoost':
        clf = xgb.XGBClassifier()  # Initialize XGBoost classifier
        clf.load_model("models/model_xgb.json")  # Load XGBoost model
    else:
        # clf = lgbm.LGBMClassifier()  # You can remove this line since LightGBM is already loaded in the else block
        clf = lgbm.Booster(model_file='models/model_lgbm.txt')  # Load LightGBM model
    return clf



def get_transformed_data(test_data=None):
    X = df_train.drop("Churn", axis=1)

    if test_data is None:
        test_data = df_test.copy()

    # Test dataset
    y_test = test_data['Churn'].values
    X_test = test_data.drop("Churn", axis=1)

    # Define numerical and categorical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = list(set(X.columns) - set(X[num_cols]))

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=cat_cols)
    X_test = pd.get_dummies(X_test, columns=cat_cols)

    # Scale numerical features
    transformer = RobustScaler()
    X[num_cols] = transformer.fit_transform(X[num_cols])
    X_test[num_cols] = transformer.transform(X_test[num_cols])

    return X_test, y_test


def make_prediction(X_test, clf):
    if isinstance(clf, xgb.XGBClassifier):
        test_pred = clf.predict_proba(X_test)[:, 1]  # probability of class 1
    elif isinstance(clf, lgbm.Booster):
        test_pred = clf.predict(X_test)
    else:
        raise ValueError("Unsupported classifier type")

    return test_pred


if st.sidebar.button('Performance on Test Dataset'):
    X_test, y_test = get_transformed_data()
    test_pred = make_prediction(X_test)
    st.write("Performance on The Test Dataset (ROC AUC Score): ", roc_auc_score(y_test, test_pred))

    # Calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(plt)

if st.sidebar.button('Prediction on Random Instance from Test Data'):
    random_test_instance = df_test.sample(n=1)
    X_test_instance, y_test_instance = get_transformed_data(random_test_instance)
    test_pred_instance = make_prediction(X_test_instance)
    st.markdown(f"Prediction on Random Instance From Test Data: {'**Churned**' if test_pred_instance[0] > 0.5 else '**Not Churned**'} (Probability: {test_pred_instance[0] : 0.2f}) ")
    st.write("Random Instance Features")
    st.dataframe(random_test_instance)

st.sidebar.markdown('## User Input')

def binning_feature(feature, value):
    bins = np.linspace(min(df_churn[feature]), max(df_churn[feature]), 4)
    if bins[0] <= value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No', 'No phone service'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No'))
    internet_service_type = st.sidebar.selectbox('Internet Service Type', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))

    payment_method = st.sidebar.selectbox('PaymentMethod', (
        'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

    # Tenure slider
    unique_tenure_values = df_churn.tenure.unique()
    min_value, max_value = min(unique_tenure_values), max(unique_tenure_values)
    tenure = st.sidebar.slider("Tenure", int(min_value), int(max_value), int(min_value), 1)

    # Monthly Charges slider
    unique_monthly_charges_values = df_churn.MonthlyCharges.unique()
    min_value, max_value = min(unique_monthly_charges_values), max(unique_monthly_charges_values)
    monthly_charges = st.sidebar.slider("Monthly Charges", min_value, max_value, float(min_value))

    # Total Charges slider
    min_value_total = monthly_charges * tenure
    max_value_total = (monthly_charges * tenure) + 100
    st.sidebar.markdown("**`TotalCharges`** = `MonthlyCharges` * `Tenure` + `Extra Cost (~100)`")
    total_charges = st.sidebar.slider("Total Charges", min_value_total, max_value_total)

    # Churn filter
    data = {'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen.lower() == 'yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service_type],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure-binned': binning_feature('tenure', tenure),
            'MonthlyCharges-binned': binning_feature('MonthlyCharges', monthly_charges),
            'TotalCharges-binned': binning_feature('TotalCharges', total_charges)
            }

    features = pd.DataFrame(data)

    return features

input_df = user_input_features()

num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = input_df.select_dtypes(include=['object']).columns

X = df_train.drop("Churn", axis=1)
user_df = input_df.copy()

X = pd.get_dummies(X, columns=cat_cols)
user_df = pd.get_dummies(user_df, columns=cat_cols)

transformer = RobustScaler()
X[num_cols] = transformer.fit_transform(X[num_cols])
user_df[num_cols] = transformer.transform(user_df[num_cols])

clf = get_classifier(classifier_name)
if st.sidebar.button('Predict'):
    test_pred = make_prediction(user_df)
    
    st.markdown(f"Prediction result: {'**Churned**' if test_pred[0] > 0.5 else '**Not Churned**'} (Probability: {test_pred[0]:.2f}) ")

    st.write("User Input Features")
    st.dataframe(input_df)