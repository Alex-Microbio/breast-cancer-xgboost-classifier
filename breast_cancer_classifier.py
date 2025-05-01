# breast_cancer_classifier.py

# =====================
# Imports
# =====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# =====================
# Load and Inspect Dataset
# =====================
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# =====================
# Preprocessing
# =====================
def preprocess_data(data):
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis'].replace({'M': 1, 'B': 0})
    return X, y

# =====================
# Split and Scale Data
# =====================
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# =====================
# Train Model
# =====================
def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# =====================
# Evaluation
# =====================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_test, y_pred

# =====================
# Visualization
# =====================
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    features_df = features_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df, palette='rainbow')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=True, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, annot_kws={"size": 8})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.text(2.5, 0, f'Precision: {precision:.2f}', ha='center', va='center', fontsize=12)
    plt.text(2.5, 0.2, f'Recall: {recall:.2f}', ha='center', va='center', fontsize=12)
    plt.text(2.5, 0.4, f'F1 Score: {f1:.2f}', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()

# =====================
# Main Execution
# =====================
def main():
    filepath = 'data/breast_cancer_data.csv'  # update path as needed

    data = load_data(filepath)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    model = train_model(X_train, y_train)
    y_test, y_pred = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, X.columns)
    plot_correlation_matrix(data)
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
