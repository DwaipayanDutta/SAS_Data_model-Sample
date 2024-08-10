import subprocess
import sys
import numpy as np
import pandas as pd
import os
from colorama import Fore, Style
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def display_script_banner():
    owner_name = 'By: Dwaipayan Dutta'
    banner = rf"""
  {Fore.YELLOW}___       _           ___                       _           
 |   \ __ _| |_ __ _   / __|___ _ _  ___ _ _ __ _| |_ ___ _ _ 
 | |) / _` |  _/ _` | | (_ / -_) ' \/ -_) '_/ _` |  _/ _ \ '_|
 |___/\__,_|\__\__,_|  \___\___|_||_\___|_| \__,_|\__\___/_|  
                                                               
    \t┌──────────────────────────────────────────────────────────────┐
    \t│                                                              │        
    \t│ {Fore.GREEN}Data Generation Code{Style.RESET_ALL}                                         │     
    \t│ {Fore.BLUE}{owner_name}{Style.RESET_ALL}                                          │
    \t│                                                              │
    \t└──────────────────────────────────────────────────────────────┘
    """
    print(banner)

class DataGenerator:
    def __init__(self, n_samples=10000, export_path='dataset.csv'):
        np.random.seed(42)
        self.n_samples = n_samples
        self.export_path = export_path
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

        # Apply constraints
        self.continuous_data['AGE'] = np.random.randint(1, 100, size=self.n_samples)  # AGE between 1 and 99
        self.continuous_data['AQB_BALANCE'] = np.random.uniform(0, 10000000, size=self.n_samples)  # AQB_BALANCE between 0 and 10,000,000
        self.continuous_data['CIBIL_SCORE'] = np.random.randint(300, 901, size=self.n_samples)  # CIBIL_SCORE between 300 and 900
        self.continuous_data['CREDIT_CARD_LIMIT'] = np.random.uniform(0, 1000000, size=self.n_samples)  # CREDIT_CARD_LIMIT between 0 and 1,000,000
        self.continuous_data['VINTAGE_DAYS'] = np.random.randint(1, 10001, size=self.n_samples)  # VINTAGE_DAYS between 1 and 10,000

        # Generate KYC_LAST_DONE_DATE between 01-01-2020 and 01-12-2023
        start_date = np.datetime64('2020-01-01')
        end_date = np.datetime64('2023-12-01')
        self.continuous_data['KYC_LAST_DONE_DATE'] = np.random.choice(pd.date_range(start_date, end_date), size=self.n_samples)

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

        # Probabilities for categorical variables
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

        # List of flag variables
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

    def export_to_csv(self):
        self.final_data.to_csv(self.export_path, index=False)
        print(f"Dataset exported to: {os.path.abspath(self.export_path)}")

class DataPipeline:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.dataset = self._generate_and_preprocess_data()

    def _generate_and_preprocess_data(self):
        # Generate dataset
        dataset = self.data_generator.generate_dataset()
        print("Initial dataset columns:", dataset.columns.tolist())  # Debugging line

        # Encode categorical variables to numerical values
        categorical_cols = dataset.select_dtypes(include='object').columns
        le = LabelEncoder()
        for col in categorical_cols:
            dataset[col] = le.fit_transform(dataset[col].astype(str))  # Convert to string before encoding

        dataset = dataset.fillna(0)
        return dataset

    def split_data(self, target_col):
        if target_col not in self.dataset.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset.")
        # Split the dataset into features and target
        X = self.dataset.drop(['ACCOUNT_NO', target_col], axis=1, errors='ignore')
        y = self.dataset[target_col]
        
        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return train_test_split(X, y, test_size=0.2, random_state=42)

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        return self.scaler.transform(X_test)

class ModelEvaluator:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        for name, model in tqdm(self.classifiers.items(), desc="Training models", unit="model"):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.results[name] = accuracy
            
            # Print confusion matrix and classification report
            print(f"\n{name} Results:")
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=1))

    def get_best_classifier(self):
        if not self.results:
            raise ValueError("No classifiers have been evaluated.")
        best_classifier_name = max(self.results, key=self.results.get)
        best_classifier_accuracy = self.results[best_classifier_name]
        return best_classifier_name, best_classifier_accuracy

class SHAPExplainer:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        
        # Explainers based on the model type
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            self.explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            self.explainer = shap.LinearExplainer(model, X_train)
        elif isinstance(model, SVC):
            self.explainer = shap.KernelExplainer(model.predict_proba, X_train)
        else:
            raise ValueError("Unsupported model type for SHAP explainer")
        
        
        try:
            self.shap_values = self.explainer.shap_values(X_test)
        except shap.ExplainerError as e:
            print(f"Error calculating SHAP values: {e}")
            # Optionally: try with additivity check turned off
            self.shap_values = self.explainer.shap_values(X_test, check_additivity=False)

    def plot_shap_values(self):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar")
        plt.show()

# Finally Jay ram ji ki 
if __name__ == "__main__":
    display_script_banner()
    
    # Generate and export dataset
    data_generator = DataGenerator(n_samples=10000)
    data_generator.generate_dataset()
    data_generator.export_to_csv()
    
    # Load dataset
    pipeline = DataPipeline(data_generator)
    
    try:
        X_train, X_test, y_train, y_test = pipeline.split_data(target_col='LI_FLAG')
    except KeyError as e:
        print(f"Error: {e}")
    
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Support Vector Classifier": SVC(probability=True)
    }
    
    evaluator = ModelEvaluator(classifiers)
    evaluator.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
    
    best_classifier_name, best_classifier_accuracy = evaluator.get_best_classifier()
    print(f"\nBest Classifier: {best_classifier_name} with Accuracy: {best_classifier_accuracy:.2f}")
    
    best_model = classifiers[best_classifier_name]
    explainer = SHAPExplainer(best_model, X_train_scaled, X_test_scaled)
    explainer.plot_shap_values()
