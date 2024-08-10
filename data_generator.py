import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=10000):
        np.random.seed(42)
        self.n_samples = n_samples
        self.continuous_data = None
        self.categorical_data = None
        self.final_data = None

    def generate_continuous_data(self):
        cont_dict = {
            'Variable': ['AGE', 'AQB_BALANCE', 'AUM', 'CIBIL_SCORE', 'CREDIT_CARD_LIMIT', 'CR_AMT_12MNTH', 'CR_CNT_12MNTH',
                         'DC_APPAREL_30DAYS_ACTV', 'DC_ECOM_30DAYS_AMT', 'DC_ECOM_30DAYS_CNT', 'DC_FOOD_30DAYS_ACTV',
                         'DC_FUEL_30DAYS_ACTV', 'DC_GROCRY_30DAYS_ACTV', 'DC_OTT_30DAYS_ACTV', 'DC_POS_30DAYS_AMT',
                         'DC_POS_30DAYS_CNT', 'DC_RECHARGE_30DAYS_ACTV', 'DC_TRAVEL_30DAYS_ACTV', 'DC_UTILITY_30DAYS_ACTV',
                         'DR_AMT_12MNTH', 'DR_CNT_12MNTH', 'DR_CR_RATIO', 'FD_COUNT', 'FD_CURRENTMONTHANR', 'INCOME_NET',
                         'KYC_LAST_DONE_DATE', 'MONTHLY_BALANCE', 'NRV', 'TOTAL_LIVE_SECURED_AMT', 'TOTAL_LIVE_UNSECURED_AMT',
                         'VINTAGE_DAYS'],
            'Mean': [45.75, 335324.5275, 1482679.66, 729.25, 432750.0, 4427092.8125, 390.0, 0.0, 747.54, 0.5, 0.0, 0.25, 0.25,
                     0.0, 1250.0, 0.75, 0.0, 0.0, 0.0, 4400780.205, 755.25, 18.4664255, 2.25, 818797.965, 637578.845, 22694.0,
                     285443.28500000003, 1427096.995, 8043387.25, 1626783.5, 2472.25],
            'Std Dev': [13.2034277, 40797470.12, 31660895.68, 145.8168632, 227937.36, 160056333.0, 405.0004491, 0.0778686,
                        55320.12, 1.1421999, 0.0979862, 0.1093538, 0.1182655, 0.081972, 6117.33, 1.1075376, 0.0678652, 0.0594367,
                        0.0595447, 163060150.0, 435.6866233, 249.0156422, 1.8768659, 2177717.88, 272956.6, 786.2686886,
                        39394146.0, 31877424.32, 24655099.6, 3300140.98, 1216.33]
        }

        data_summary = pd.DataFrame(cont_dict)
        data_summary = data_summary[~data_summary['Variable'].isin(['FD_FLAG', 'GI_FLAG', 'HEALTH_FLAG', 'LI_FLAG', 'MASS_FLAG',
                                                                   'MF_FLAG', 'NR_FLAG'])]

        # Continuous data
        self.continuous_data = pd.concat([
            pd.DataFrame({row['Variable']: np.random.gamma(row['Mean'], row['Std Dev'], size=self.n_samples)})
            for _, row in data_summary.iterrows()
        ], axis=1)

    def generate_categorical_data(self):
        cat_variables = {
            'ACCOUNT_TYPE': ['SAVINGS', 'CURRENT'],
            'CUSTOMER_TAG': ['RETAIL', 'INSTITUTIONAL'],
            'EDUCATION_LEVEL': ['Below 10th', 'UG', 'Graduate', 'PG'],
            'GENDER': ['MALE', 'FEMALE'],
            'INTERNET_BANKING_USAGE': ['Y', 'N'],
            'MARITAL_STATUS': ['Married', 'Single/Unmarried', 'Divorced/Separated'],
            'NOMINEE_AVAILABLE_FLAG': ['Y', 'N'],
            'RM_ALLOCATED_FLAG': ['Y', 'N'],
            'OCCUPATION': ['SELF EMPLOYED', 'SALARIED', 'BUSINESS OWNER', 'NON-EARNER'],
            'STATE': ['Maharashtra', 'Gujarat', 'West Bengal', 'Uttar Pradesh', 'Tamil Nadu', 'Haryana', 'Rajasthan', 'Punjab',
                      'Delhi', 'Madhya Pradesh', 'Out of India', 'Andhra Pradesh', 'Karnataka', 'Telangana', 'Odisha', 'Kerala',
                      'Bihar', 'Assam', 'Chhattisgarh', 'Jharkhand', 'Uttarakhand', 'Himachal Pradesh', 'Jammu and Kashmir',
                      'Chandigarh', 'Goa', 'Manipur', 'Nagaland', 'Tripura', 'Puducherry', 'Dadra and Nagar Haveli and Daman and Diu',
                      'Sikkim', 'Arunachal Pradesh', 'Meghalaya', 'Mizoram', 'Andaman and Nicobar']
        }

        # Prob
        prob = {
            'ACCOUNT_TYPE': [0.80, 0.20],
            'CUSTOMER_TAG': [0.5, 0.5],
            'EDUCATION_LEVEL': [0.25, 0.25, 0.25, 0.25],
            'GENDER': [0.6, 0.4],
            'INTERNET_BANKING_USAGE': [0.4, 0.6],
            'MARITAL_STATUS': [0.6, 0.35, 0.05],
            'NOMINEE_AVAILABLE_FLAG': [0.95, 0.05],
            'RM_ALLOCATED_FLAG': [0.98, 0.02],
            'OCCUPATION': [0.404, 0.339, 0.152, 0.105],
            'STATE': [0.119496018, 0.089888389, 0.07827775, 0.072670547, 0.064865747, 0.062475027, 0.050854399, 0.049988679,
                      0.046685624, 0.043482459, 0.036553368, 0.033316907, 0.03284409, 0.031895128, 0.030473349, 0.029424496,
                      0.028928372, 0.020664074, 0.018809435, 0.015855997, 0.007232093, 0.006562825, 0.005510642, 0.005430729,
                      0.005330838, 0.00238739, 0.001664846, 0.001534988, 0.001481713, 0.001328547, 0.001288591, 0.001175382,
                      0.00107882, 0.000526091, 1.66485E-05]
        }

        # List All flags 
        flag_variables = ['FD_FLAG', 'GI_FLAG', 'HEALTH_FLAG', 'LI_FLAG', 'MASS_FLAG', 'MF_FLAG', 'NR_FLAG']

        for flag_var in flag_variables:
            cat_variables[flag_var] = ['Y', 'N']
            prob[flag_var] = [0.5, 0.5]

        # Categorical data
        self.categorical_data = pd.DataFrame({col: np.random.choice(cat_variables[col], self.n_samples, p=prob[col]) for col in cat_variables})
        self.categorical_data['ACCOUNT_NO'] = self.categorical_data.index

    def combine_data(self):
        # Combine categorical and continuous data
        self.final_data = pd.concat([self.categorical_data, self.continuous_data], axis=1)

    def generate_dataset(self):
        self.generate_continuous_data()
        self.generate_categorical_data()
        self.combine_data()
        return self.final_data

# Calling function 
data_generator = DataGenerator(n_samples=10000)
dataset = data_generator.generate_dataset()
print(dataset.head())


# Model Code for testing :
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Generate dataset
data_generator = DataGenerator(n_samples=10000)
dataset = data_generator.generate_dataset()

# Encode categorical variables to numerical values
categorical_cols = dataset.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])

# Fill missing values
dataset = dataset.fillna(0)

# Split the dataset into features and target
X = dataset.drop(['ACCOUNT_NO', 'LI_FLAG'], axis=1)
y = dataset['LI_FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers to evaluate
classifiers = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=2000),  # Increased iterations
    "Support Vector Classifier": SVC(probability=True)
}

# Store results for comparison
results = {}

# Train and evaluate each classifier
for name, model in classifiers.items():
    if name in ["Logistic Regression"]:
        model.fit(X_train_scaled, y_train)  # Use scaled data for Logistic Regression
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)  # Use original data for other classifiers
        y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    # Print confusion matrix and classification report
    print(f"\n{name} Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    #print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))

# Determine the best classifier
best_classifier_name = max(results, key=results.get)
best_classifier_accuracy = results[best_classifier_name]

print(f"\nBest Classifier: {best_classifier_name} with Accuracy: {best_classifier_accuracy:.2f}")

# Explain the best model's predictions using SHAP values
best_model = classifiers[best_classifier_name]
explainer = shap.Explainer(best_model, X_train_scaled if best_classifier_name == "Logistic Regression" else X_train)
shap_values = explainer(X_test_scaled if best_classifier_name == "Logistic Regression" else X_test)

# Visualize the SHAP values
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled if best_classifier_name == "Logistic Regression" else X_test, plot_type="bar")
plt.show()
