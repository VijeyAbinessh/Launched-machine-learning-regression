# Launched-machine-learning-regression

# 💼 Expected CTC Prediction using Linear Regression

This project applies a simple Linear Regression model to predict the **Expected Cost to Company (CTC)** of applicants using numeric features from a given dataset.

## 📁 Dataset
- Loaded from: `expected_ctc.csv`
- Preprocessed to:
  - Remove columns: `IDX`, `Applicant_ID`
  - Drop rows with missing values
  - Retain only numeric features
  - Target variable: `Expected_CTC`

## 📊 Features
The model uses all available **numeric** features as input to predict `Expected_CTC`.

## 🧪 Steps Performed

1. **Data Cleaning**  
   - Dropped unnecessary ID columns  
   - Removed any rows with missing values

2. **Feature Selection**  
   - Retained only columns with `int64` and `float64` datatypes

3. **Train-Test Split**  
   - 80% training, 20% testing using `train_test_split`

4. **Feature Scaling**  
   - Standardized features using `StandardScaler`

5. **Modeling**  
   - Applied `LinearRegression` from scikit-learn

6. **Evaluation Metrics**  
   - **MAE** (Mean Absolute Error)  
   - **RMSE** (Root Mean Squared Error)  
   - **R²** (Coefficient of Determination)

7. **Visualization**  
   - Scatter plot comparing actual vs predicted CTC values

## 📈 Results (Sample Output)
```
📊 Linear Regression Results (Numeric Only):
MAE : 12.45
RMSE: 15.30
R²  : 0.7432
```

## 🛠 Dependencies

Install required libraries via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🔍 Visualization

- **Scatter plot** illustrates the accuracy of model predictions vs actual `Expected_CTC`
- A red dotted line highlights the ideal fit (perfect prediction)

## 🚀 How to Run

1. Place `expected_ctc.csv` in the designated directory.
2. Update the file path in the code if needed.
3. Run the Python script to view evaluation results and visualization.

