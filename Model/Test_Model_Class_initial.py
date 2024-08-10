from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
import pandas as pd

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
        for name, model in self.classifiers.items():
            print(f"Training {name}...")
            if name == "Logistic Regression":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
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
        self.explainer = shap.Explainer(model, X_train)
        self.shap_values = self.explainer(X_test)

    def plot_shap_values(self):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar")
        plt.show()

data_generator = DataGenerator(n_samples=10000)
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



class SHAPExplainer:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        
        # Use different explainers based on the model type
        if isinstance(model, (XGBClassifier, RandomForestClassifier)):
            self.explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            self.explainer = shap.LinearExplainer(model, X_train)
        elif isinstance(model, SVC):
            self.explainer = shap.KernelExplainer(model.predict_proba, X_train)
        else:
            raise ValueError("Unsupported model type for SHAP explainer")
        
        # Calculate SHAP values
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
best_model = classifiers[best_classifier_name]
explainer = SHAPExplainer(best_model, X_train_scaled, X_test_scaled)
explainer.plot_shap_values()


