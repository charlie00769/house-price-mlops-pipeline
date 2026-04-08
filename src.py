import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# ✅ FIXED HERE (H capital)
df = pd.read_csv("data/Housing.csv")
df = df.dropna()

X = df[['area', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("r2_score", score)

    mlflow.sklearn.log_model(model, "model")

    print("Model trained with MLflow!")