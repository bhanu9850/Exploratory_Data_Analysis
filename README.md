# Exploratory_Data_Analysis
# ML Data Preprocessing and Exploratory Data Analysis (EDA)

This project demonstrates how to clean, preprocess, and prepare a synthetic dataset for machine learning using Python and pandas. It includes detailed steps for handling missing data, converting types, encoding categorical variables, and evaluating a regression model.

---

## Files & Structure
project-folder/
├── data/
│ ├── synthetic_ml_dataset.csv # Original dataset
│ ├── preprocessed_dataset.csv # Cleaned and encoded dataset
├── ML Data Preprocessing.py # Full preprocessing + model code
├── README.md



---

##  Key Features

-  Missing value handling (`NaN`, invalid strings)
-  Data type conversion (`object` to numeric/datetime)
-  Encoding categorical variables using one-hot encoding
-  Exploratory data analysis (EDA) summary with `.info()`, `.describe()`, `.isnull().sum()`
-  Regression model with `RandomForestRegressor`
-  Model evaluation using MAE, MSE, RMSE, and R² score

---

##  Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- Seaborn / Matplotlib (optional for visualizations)

---

##  How to Run

1. Clone the repository
2. Open `ML Data Preprocessing.py` in your IDE (Jupyter, VS Code, etc.)
3. Run the notebook/script step-by-step
4. The cleaned dataset will be saved as `preprocessed_dataset.csv`

---

##  Output Preview

Example of model evaluation:

MAE: 0.58
MSE: 0.64
RMSE: 0.80
R² Score: 0.72

##  Author

**Kuruva Bhanu Prakash**  
-  [LinkedIn](https://www.linkedin.com/in/kuruva-bhanu-prakash-b681a6204/)  
-  [GitHub](https://github.com/bhanu9850)




