
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\vijey abinessh\Desktop\expected_ctc.csv")

# Drop unnecessary columns
df.drop(columns=["IDX", "Applicant_ID"], inplace=True)

# Drop all rows with missing values
df.dropna(inplace=True)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Separate features and target
X = numeric_df.drop(columns=["Expected_CTC"])
y = numeric_df["Expected_CTC"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Output results
print("\n📊 Linear Regression Results (Numeric Only):")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Expected CTC")
plt.ylabel("Predicted Expected CTC")
plt.title("Actual vs Predicted CTC")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.show()
