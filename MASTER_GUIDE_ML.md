# 🤖 The Universal Machine Learning Mastery Guide
**Author:** Saleem Ali
**Curriculum:** Al Nafi Machine Learning (Labs 1–50)
**Purpose:** Personal Code Reference & Execution Manual

---

## 📋 Lab Index (Machine Learning)
*Click the Module link to jump to the code.*

| Labs | Topic | Module |
| :--- | :--- | :--- |
| **1–9** | Variables, Loops, Functions, Data Structures | [Module 1: Python Fundamentals](#-module-1-python-fundamentals) |
| **10–18** | NumPy Arrays, Pandas DataFrames, Visualization | [Module 2: Data Science Toolkit](#-module-2-data-science-toolkit) |
| **19–20** | Scaling, Encoding, Train/Test Splits | [Module 3: Preprocessing](#-module-3-data-preprocessing) |
| **21–30** | Linear/Logistic Regression, KNN, SVM | [Module 4: ML Algorithms](#-module-4-machine-learning-algorithms) |
| **31–40** | Decision Trees, Random Forests, Ensembles | [Module 4: ML Algorithms](#-module-4-machine-learning-algorithms) |
| **41–50** | Clustering (K-Means), PCA, Capstone | [Module 5: Unsupervised & Capstone](#-module-5-unsupervised--capstone) |

---

## 🟢 Module 1: Python Fundamentals

### Lab 1: Environment Setup
* **Goal:** Verify Python & Libraries.
* **Script:** `01_check_env.py`
```python
import sys
import pandas as pd
import sklearn
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")

```

### Lab 2: Variables & Data Types

* **Goal:** Store data types.
* **Script:** `02_variables.py`

```python
name = "Ali"       # String
age = 25           # Integer
height = 5.9       # Float
is_student = True  # Boolean
print(f"{name} is {age} years old.")

```

### Lab 3: Control Flow (If/Else)

* **Goal:** Logic gates.
* **Script:** `03_logic.py`

```python
score = 85
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
else:
    print("Grade: C")

```

### Lab 4: Loops (For/While)

* **Goal:** Iteration.
* **Script:** `04_loops.py`

```python
# Loop through a list
fruits = ["Apple", "Banana", "Cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# While loop
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1

```

### Lab 5: Data Structures (Lists & Dicts)

* **Goal:** Organize data.
* **Script:** `05_structures.py`

```python
# List (Ordered)
numbers = [1, 2, 3, 4, 5]
numbers.append(6)

# Dictionary (Key-Value)
student = {"id": 101, "name": "Saleem", "course": "AI"}
print(student["name"])

```

### Lab 6: Functions

* **Goal:** Reusable code blocks.
* **Script:** `06_functions.py`

```python
def calculate_area(width, height):
    return width * height

print(f"Area: {calculate_area(10, 5)}")

```

---

## 🔵 Module 2: Data Science Toolkit

### Lab 10: NumPy Basics

* **Goal:** Fast math on arrays.
* **Script:** `10_numpy_arrays.py`

```python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(f"Mean: {np.mean(arr)}")
print(f"Squared: {arr ** 2}")

```

### Lab 11: Pandas DataFrames

* **Goal:** Excel-like data manipulation.
* **Script:** `11_pandas_intro.py`

```python
import pandas as pd
data = {'Name': ['Ali', 'Sara'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df.head())

```

### Lab 12: Loading Data

* **Goal:** Read CSV files.
* **Script:** `12_load_csv.py`

```python
# df = pd.read_csv('data.csv')
# print(df.info())  # Check for nulls
# print(df.describe()) # Statistics

```

### Lab 15: Matplotlib Visualization

* **Goal:** Line and Bar charts.
* **Script:** `15_plotting.py`

```python
import matplotlib.pyplot as plt
x = [1, 2, 3]
y = [10, 20, 30]
plt.plot(x, y)
plt.title("Simple Plot")
plt.show()

```

### Lab 16: Seaborn Visualization

* **Goal:** Statistical plots (Heatmaps).
* **Script:** `16_seaborn.py`

```python
import seaborn as sns
# sns.heatmap(df.corr(), annot=True)
# plt.show()

```

---

## 🟠 Module 3: Data Preprocessing

### Lab 19: Handling Missing Values

* **Goal:** Clean dirty data.
* **Script:** `19_cleaning.py`

```python
# Fill missing age with mean
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# Drop rows with missing values
# df.dropna(inplace=True)

```

### Lab 20: Encoding Categorical Data

* **Goal:** Text -> Numbers.
* **Script:** `20_encoding.py`

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# df['City_Code'] = le.fit_transform(df['City'])

```

### Lab 21: Train/Test Split

* **Goal:** Separate data for validation.
* **Script:** `21_splitting.py`

```python
from sklearn.model_selection import train_test_split
X = [[1], [2], [3], [4]] # Features
y = [0, 0, 1, 1]         # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

```

---

## 🟣 Module 4: Machine Learning Algorithms

### Lab 22: Linear Regression

* **Goal:** Predict a number (e.g., Price).
* **Script:** `22_linear_reg.py`

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Prediction: {model.predict([[5]])}")

```

### Lab 23: Logistic Regression

* **Goal:** Predict a Class (Yes/No).
* **Script:** `23_logistic_reg.py`

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Class: {model.predict(X_test)}")

```

### Lab 24: K-Nearest Neighbors (KNN)

* **Goal:** Classification by proximity.
* **Script:** `24_knn.py`

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

```

### Lab 25: Support Vector Machines (SVM)

* **Goal:** Finding the best boundary line.
* **Script:** `25_svm.py`

```python
from sklearn.svm import SVC
model = SVC(kernel='linear') # Or 'rbf'
model.fit(X_train, y_train)

```

### Lab 31: Decision Trees

* **Goal:** Flowchart-based learning.
* **Script:** `31_decision_tree.py`

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

```

### Lab 32: Random Forest

* **Goal:** Ensemble of Trees (Better accuracy).
* **Script:** `32_random_forest.py`

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

```

### Lab 35: Model Evaluation

* **Goal:** Check Accuracy & Confusion Matrix.
* **Script:** `35_evaluation.py`

```python
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

```

---

## ⚫ Module 5: Unsupervised & Capstone

### Lab 41: K-Means Clustering

* **Goal:** Grouping data without labels.
* **Script:** `41_kmeans.py`

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(f"Labels: {kmeans.labels_}")

```

### Lab 42: PCA (Dimensionality Reduction)

* **Goal:** Reducing features to speed up training.
* **Script:** `42_pca.py`

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

```

### Lab 50: The ML Pipeline (Capstone)

* **Goal:** End-to-End Workflow.
* **Script:** `50_pipeline.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Define Pipeline (Scale -> Train)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# 2. Train
pipeline.fit(X_train, y_train)

# 3. Predict
print(f"Pipeline Accuracy: {pipeline.score(X_test, y_test)}")

```

```

```
