import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

def regression_summary_and_plots(data, selected_features):
    df = data[selected_features + ['Duration_Hours']].dropna()

    # Separate target and features
    X = df[selected_features]
    y = df['Duration_Hours']

    # Identify types for preprocessing
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessor pipeline with backward-compatible OneHotEncoder
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols) # sparse_output=False
    ])

    # Transform features
    X_encoded = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Add constant for statsmodels
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit model
    model = sm.OLS(y_train, X_train_const).fit()

    # Predict and calculate residuals
    y_pred = model.predict(X_test_const)
    residuals = y_test - y_pred

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # QQ Plot
    sm.qqplot(residuals, line='s', ax=axes[0])
    axes[0].set_title('QQ Plot')

    # Residual Histogram
    axes[1].hist(residuals, bins=30, color='skyblue', edgecolor='black')
    axes[1].set_title('Histogram of Residuals')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')

    # Residuals vs Predictions
    axes[2].scatter(y_pred, residuals, color='salmon', edgecolor='k')
    axes[2].axhline(y=0, color='black', linestyle='--')
    axes[2].set_title('Residuals vs Predictions')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

    print(model.summary())
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
