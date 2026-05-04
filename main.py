import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(ticker="AAPL"):
    """Download stock data"""
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False)
    return data


def create_features(data):
    """Create technical indicators"""
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(10).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)

    return data


def create_target(data):
    """Create prediction target"""
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data


def prepare_data(data):
    """Clean and split data (time-series safe)"""
    data = data.dropna()

    features = ['MA10', 'MA50', 'Daily_Return', 'Volatility', 'Momentum']
    X = data[features]
    y = data['Target']

    split = int(len(data) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")

    return predictions


def plot_confusion_matrix(y_test, predictions):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"],
                yticklabels=["Down", "Up"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_predictions(y_test, predictions):
    """Plot actual vs predicted"""
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.legend()
    plt.title("Prediction vs Actual (First 100 Points)")
    plt.tight_layout()
    plt.show()


def main():
    print("📊 Stock Prediction Model Running...\n")

    data = load_data()
    data = create_features(data)
    data = create_target(data)

    X_train, X_test, y_train, y_test = prepare_data(data)

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(y_test, predictions)
    plot_predictions(y_test, predictions)


if __name__ == "__main__":
    main()