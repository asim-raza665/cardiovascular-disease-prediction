# Cardiovascular Disease Prediction Using Lifestyle and Health Data

A machine learning project that predicts the presence of cardiovascular disease using lifestyle and clinical health data, including age, blood pressure, cholesterol, glucose level, and physical activity.

## Dataset
Real-world cardiovascular health dataset (`Risk Factors for Cardiovascular Heart Disease.csv`) containing patient records with demographic, clinical, and lifestyle features.

## Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (modeling & evaluation)
- Joblib (model persistence)

## Workflow
1. **Data Cleaning** — handled missing values, removed unrealistic blood pressure readings, converted age from days to years.
2. **Feature Engineering** — created age groups, dropped irrelevant ID columns, one-hot encoded categorical features.
3. **Exploratory Data Analysis** — visualized age distribution, cardio disease distribution, blood pressure trends, cholesterol and activity levels against disease outcome.
4. **Model Training** — trained and compared three classifiers:
   - Decision Tree
   - Logistic Regression
   - Random Forest
5. **Evaluation** — assessed models using accuracy, classification report, and confusion matrix. Random Forest was selected as the best-performing model.

## Results
The Random Forest model achieved the best overall performance among the three classifiers, evaluated using accuracy score, precision, recall, F1-score, and confusion matrix analysis.

## How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn joblib
python cardiovascular_prediction.py
```
## Author
Asim Raza
GitHub: https://github.com/asim-raza665
