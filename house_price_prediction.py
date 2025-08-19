import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load the training data
# -------------------------------
train = pd.read_csv("train.csv")  # Make sure train.csv is in the same folder

# -------------------------------
# 2. Select features and target
# -------------------------------
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
X = train[features]
y = train["SalePrice"]

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train Linear Regression model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate model
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# -------------------------------
# 6. Predict on test.csv and save submission
# -------------------------------
test = pd.read_csv("test.csv")
X_test_data = test[features]

# Handle missing values
X_test_data = X_test_data.fillna(0)

test_predictions = model.predict(X_test_data)

# Save submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("\n✅ Predictions saved to submission.csv")

# -------------------------------
# 7. Predict using user input
# -------------------------------
print("\n🔮 Enter custom house details for price prediction:")

area = float(input("Enter square footage (GrLivArea): "))
bedrooms = int(input("Enter number of bedrooms: "))
full_bath = int(input("Enter number of full bathrooms: "))
half_bath = int(input("Enter number of half bathrooms: "))

custom_house = pd.DataFrame({
    "GrLivArea": [area],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [full_bath],
    "HalfBath": [half_bath]
})

predicted_price = model.predict(custom_house)[0]
print(f"\n💰 Predicted Price for your house: ${predicted_price:,.2f}")

# -------------------------------
# 8. Visualization (optional)
# -------------------------------
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
