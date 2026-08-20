# ============================================================
# Cardiovascular Disease Prediction Using Lifestyle and Health Data
# CSC-102L Programming Fundamentals - Fall 2025
#Asim Raza(2025-BSCPE-138)
# Instructor: Dr. Bilal Ahmad
# UET Lahore, Faisalabad Campus
# ============================================================

# ============================================================
# Block 01: Load Dataset
# ============================================================
import pandas as pd

df = pd.read_csv("Risk Factors for Cardiovascular Heart Disease.csv", sep=";")
print(df.head())
print(df.info())
print(df.describe())

# ============================================================
# Block 02: Handling Missing Values
# ============================================================
print(df.isnull().sum())
df = df.dropna()

# ============================================================
# Block 03: Feature Engineering
# ============================================================

# Convert age from days to years
df["age_years"] = (df["age"] / 365).astype(int)
df.drop("age", axis=1, inplace=True)

# Drop unnecessary ID columns
df.drop(["index", "id"], axis=1, inplace=True)

# ============================================================
# Block 04: Cleaning Blood Pressure Data
# ============================================================
df = df[
    (df['ap_hi'].between(60, 250)) &   # realistic systolic range
    (df['ap_lo'].between(40, 160)) &   # realistic diastolic range
    (df['ap_hi'] > df['ap_lo'])        # systolic must always be higher than diastolic
].copy()

# ============================================================
# Block 4.1: Visualization - Age Distribution
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(df["age_years"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ============================================================
# Block 4.2: Visualization - Target Distribution (Cardio)
# ============================================================
sns.countplot(data=df, x="cardio")
plt.title("Cardio Distribution")
plt.show()

# ============================================================
# Block 4.3: Visualization - Systolic Blood Pressure vs Age
# ============================================================
sns.scatterplot(data=df, x="age_years", y="ap_hi", hue="cardio")
plt.title("Systolic BP vs Age")
plt.show()

# ============================================================
# Block 4.4: Visualization - Diastolic Blood Pressure by Cardio
# ============================================================
sns.boxplot(data=df, x="cardio", y="ap_lo")
plt.title("Diastolic BP by Cardio")
plt.show()

# ============================================================
# Block 4.5: Visualization - Cholesterol Level vs Cardio
# ============================================================
sns.countplot(data=df, x="cholesterol", hue="cardio")
plt.title("Cholesterol vs Cardio")
plt.show()

# ============================================================
# Block 4.6: Visualization - Physical Activity vs Cardio
# ============================================================
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="active", hue="cardio")
plt.title("Physical Activity vs Cardio")
plt.show()

# ============================================================
# Block 4.7: Creation of Age Groups
# ============================================================
df["Age Group"] = pd.cut(
    df["age_years"],
    bins=[0, 35, 45, 60, 100],
    labels=["Young", "Adult", "Middle Age", "Old"]
)

# ============================================================
# Block 05: Visualization - Age Group vs Cardio
# ============================================================
sns.countplot(data=df, x="Age Group", hue="cardio")
plt.title("Cardio Across Age Group")
plt.show()

# ============================================================
# Block 06: Encoding Categorical Data
# ============================================================
df = pd.get_dummies(df, drop_first=True)

# ============================================================
# Block 07: Train-Test Split
# ============================================================
from sklearn.model_selection import train_test_split

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ============================================================
# Block 08: Model Training
# ============================================================

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ============================================================
# Block 09: Save and Load Models
# ============================================================
import joblib

joblib.dump(dt, "decision_tree_model.pkl")
joblib.dump(lr, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")

# Load best model (Random Forest)
model = joblib.load("random_forest_model.pkl")

# ============================================================
# Block 10: Prediction on New Data
# ============================================================
y_pred = model.predict(X_test)
print(y_pred[:10])

# ============================================================
# Block 11: Classification Performance Metrics
# ============================================================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# Block 12: Actual VS Predicted Probability Analysis
# ============================================================
y_prob = model.predict_proba(X_test)[:, 1]
plt.scatter(y_test, y_prob)
plt.xlabel("Actual Cardio (0 = No, 1 = Yes)")
plt.ylabel("Predicted Probability of Cardio")
plt.title("Actual vs Predicted Cardio Probability")
plt.show()
