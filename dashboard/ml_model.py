# ml_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(df, indicator='CO2'):
    """
    Train linear regression model for a specific indicator.
    Returns model, mse, X_test, y_test, predictions
    """
    # Filter indicator
    df = df[df['Indicator'].str.contains(indicator, case=False)]
    
    if df.empty or len(df) < 2:
        # Too few samples to train or split
        return None, None, None, None, None

    X = df[['Year']]
    y = df['Value']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)

    return model, mse, X_test, y_test, pred