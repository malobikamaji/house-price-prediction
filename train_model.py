import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
data = pd.read_csv("house_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# 2. Separate input features and target
X = data.drop("price", axis=1)
y = data["price"]


# 3. Identify categorical columns
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]


# 4. Convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# 5. Create the machine learning model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# 6. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 8. Train the model
print("\nTraining the model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# 9. Make predictions
y_pred = pipeline.predict(X_test)


# 10. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n----- MODEL PERFORMANCE -----")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# 11. Save the trained model
joblib.dump(pipeline, "house_price_model.pkl")

print("\nModel saved as house_price_model.pkl")