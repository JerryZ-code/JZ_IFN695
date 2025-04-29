import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)
import matplotlib.pyplot as plt

def train_and_evaluate_rf(df, train_model=True, model=None, test_size=0.2, random_state=42):

    # --- 1. Select Features and Target
    features = ['Recall_Day_Hours', 'Recall_Night_Hours', # numerical columns
                'Project Work?', 'DNSP Aware?', 'Generator Aware?', 'Inter-Regional', # boolean columns
                'Region', 'NSP', 'Asset Type', 'Reason', 'Status_Code', 'Status_Description' # categorical coumns
                ] 
    target = 'Duration_Category'

    X = df[features].copy()
    y = df[target]

    # --- 2. Encode Categorical Features
    for col in ['Region', 'NSP', 'Asset Type', 'Reason', 'Status_Code', 'Status_Description']:
        X[col] = LabelEncoder().fit_transform(X[col])

    # --- 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- 4. Train Model (if needed)
    if train_model or model is None:
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)

    # --- 5. Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # --- 6. Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # --- 7. Get present labels (classes that actually exist)
    full_label_order = ['0-6 hr', '6-12 hr', '12-24 hr', '24hr-1wk', '1w-1mo', '1+mo']
    present_labels = [label for label in full_label_order if label in y_test.unique()]


    # --- 8. Classification Report
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, labels=present_labels, target_names=present_labels))

    # --- 9. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    axes[0].barh(np.array(feature_names)[indices], importances[indices], color='goldenrod')
    axes[0].invert_yaxis()
    axes[0].set_title("Feature Importances")
    axes[0].set_xlabel("Importance")

    # Confusion Matrices
    cm_train = confusion_matrix(y_train, y_train_pred, labels=present_labels)
    cm_test = confusion_matrix(y_test, y_test_pred, labels=present_labels)

    disp_train = ConfusionMatrixDisplay(cm_train, display_labels=present_labels)
    disp_train.plot(ax=axes[1], cmap='viridis', xticks_rotation=45, values_format='d', colorbar=False)
    axes[1].set_title(f"Train Confusion Matrix\n(Acc: {train_acc:.2%})")

    disp_test = ConfusionMatrixDisplay(cm_test, display_labels=present_labels)
    disp_test.plot(ax=axes[2], cmap='viridis', xticks_rotation=45, values_format='d', colorbar=False)
    axes[2].set_title(f"Test Confusion Matrix\n(Acc: {test_acc:.2%})")

    plt.tight_layout()
    plt.show()

    return model  # Return model for possible reuse