import sys
import subprocess

try:
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "scikit-learn", "seaborn", "matplotlib"])
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Handle Missing Values for Categorical Variables
def fill_missing_categorical(df, categorical_columns):
    """Fill missing values in categorical columns with 'Not Available'."""
    for column in categorical_columns:
        df[column].fillna('Not Available', inplace=True)

## Split Data into Train, Test, and Validation Sets
def split_dataset(df, target_column, test_size=0.4, random_state=42):
    """Split dataset into training, testing, and validation sets."""
    train, test_valid = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_column])
    test, valid = train_test_split(test_valid, test_size=0.5, random_state=random_state, stratify=test_valid[target_column])
    return train, test, valid

## Impute Missing Values in Numerical Columns
def impute_numerical_values(train, test, valid, numerical_columns):
    """Impute missing values in numerical columns using KNN."""
    imputer = KNNImputer(n_neighbors=5)
    train[numerical_columns] = imputer.fit_transform(train[numerical_columns])
    test[numerical_columns] = imputer.transform(test[numerical_columns])
    valid[numerical_columns] = imputer.transform(valid[numerical_columns])
    return train, test, valid

## Scale Numerical Features
def scale_numerical_features(train, test, valid, numerical_columns):
    """Standardize numerical features using StandardScaler."""
    scaler = StandardScaler()
    train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
    test[numerical_columns] = scaler.transform(test[numerical_columns])
    valid[numerical_columns] = scaler.transform(valid[numerical_columns])
    return train, test, valid

## Encode Categorical Features
def encode_categorical_features(train, test, valid, categorical_columns):
    """Apply label encoding to categorical features."""
    encoder = LabelEncoder()
    for column in categorical_columns:
        train[column] = encoder.fit_transform(train[column])
        test[column] = encoder.transform(test[column])
        valid[column] = encoder.transform(valid[column])
    return train, test, valid

## Visualize Performance Metrics
def visualize_confusion_matrix(y_true, y_pred, model_name):
    """Plot the confusion matrix for model evaluation."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

## Model Evaluation Class
class ModelEvaluator:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train models and evaluate performance."""
        for name, model in self.classifiers.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.results[name] = accuracy

            # Print performance metrics
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            visualize_confusion_matrix(y_test, y_pred, name)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=1))

## Main Function to Execute the Workflow
def main(df, target_column, categorical_columns, numerical_columns):
    """Execute the data preprocessing and model evaluation workflow."""
    # Fill missing values in categorical features
    fill_missing_categorical(df, categorical_columns)

    # Split the dataset
    train, test, valid = split_dataset(df, target_column)

    # Impute missing values in numerical columns
    train, test, valid = impute_numerical_values(train, test, valid, numerical_columns)

    # Scale numerical features
    train, test, valid = scale_numerical_features(train, test, valid, numerical_columns)

    # Encode categorical features
    train, test, valid = encode_categorical_features(train, test, valid, categorical_columns)

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }
    
    # Evaluate models
    evaluator = ModelEvaluator(classifiers=classifiers)
    evaluator.train_and_evaluate(X_train=train[categorical_columns + numerical_columns], 
                                  y_train=train[target_column], 
                                  X_test=test[categorical_columns + numerical_columns], 
                                  y_test=test[target_column])
