#  AI & Machine Learning Lab Portfolio

## => Project Overview
This repository hosts a comprehensive collection of **50 practical labs** completed during the **Artificial Intelligence and Machine Learning certification** at **Al Nafi International College**.

It serves as a professional portfolio demonstrating end-to-end competency in the Machine Learning lifecycle, ranging from foundational Python scripting and Data Engineering to deploying Deep Learning models and NLP systems.

**Tech Stack:** Python 3.x, Scikit-Learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn, NLTK, BeautifulSoup.

---

## => Repository Structure

The labs are organized into **6 dedicated modules** that reflect the data science workflow:

```text
/Al-Nafi-ML-Portfolio
│
├── 01_Python_Fundamentals/      # Core syntax, I/O, and Logic
├── 02_Data_Science_Toolkit/     # NumPy, Pandas, Visualization & Web Scraping
├── 03_Data_Preprocessing/       # Cleaning, Scaling, Pipelines & Engineering
├── 04_Machine_Learning_Models/  # Regression, Classification & Ensembles
├── 05_Deep_Learning_and_NLP/    # Neural Networks (ANN/CNN/RNN) & Text Processing
└── 06_Capstone_Project/         # End-to-End ML Workflow

```

---

## => Module Details

### 1. Python Fundamentals

*Focus: Core programming logic required for ML engineering.*

* **Environment & Syntax:** Virtual environments, variables, and control structures (If/Else, Loops).
* **Modular Code:** Writing reusable functions with parameters and return values.
* **Data Structures:** Efficient use of Lists, Tuples, Dictionaries, and Sets for data storage.
* **I/O Operations:** Advanced file handling (reading/writing `txt` and `csv` files) and DateTime parsing.

### 2. Data Science Toolkit

*Focus: Data manipulation and exploratory analysis.*

* **Numerical Computing:** Array vectorization, broadcasting, slicing, and reshaping 1D/2D arrays with **NumPy**.
* **Data Analysis:** DataFrame manipulation, Series creation, handling missing values, and EDA with **Pandas**.
* **Visualization:** Creating Line, Bar, and Scatter plots with **Matplotlib** and statistical plots (Box/Violin) with **Seaborn**.
* **Web Scraping:** Extracting dataset features from HTML using `BeautifulSoup` and `requests`.
* **Tree Visualization:** Visualizing Decision Tree logic using `graphviz`.

### 3. Data Preprocessing & Engineering

*Focus: Transforming raw data for model consumption.*

* **Feature Scaling:** Implementation of `StandardScaler` to normalize feature distributions.
* **Encoding:** Handling categorical variables with `LabelEncoder` and `OneHotEncoder`.
* **Feature Engineering:** Creating interaction features and analyzing feature importance.
* **Pipelines:** Building `scikit-learn` Pipelines with Custom Transformers for automated workflows.
* **Augmentation & Cleaning:** Regex pattern matching and basic data augmentation (image rotation, text synonyms).

### 4. Machine Learning Algorithms

*Focus: Supervised learning implementation using Scikit-Learn.*

* **Regression:** Simple Linear Regression and model evaluation (MSE, R2 Score).
* **Classification:** * Logistic Regression (Binary Classification).
* k-Nearest Neighbors (k-NN).
* Decision Trees (Entropy vs. Gini Index).
* Support Vector Machines (SVM) with Kernel tuning (Linear, RBF, Poly).


* **Ensemble Methods:** * Bagging (Bootstrap Aggregating).
* Boosting (AdaBoost).
* Random Forests (Feature Importance).


* **Optimization:** Hyperparameter tuning using `GridSearchCV` and k-Fold Cross-Validation.
* **Persistence:** Saving and loading trained models using `joblib`.

### 5. Deep Learning & NLP

*Focus: Advanced AI architectures and text processing.*

* **Neural Networks:** Building Sequential models with Keras (Dense layers, ReLU/Sigmoid activations).
* **Computer Vision:** Convolutional Neural Networks (CNNs) for image classification (MNIST dataset).
* **Sequential Data:** Recurrent Neural Networks (RNNs) for time-series/sequence prediction.
* **Training Analysis:** Visualizing Loss/Accuracy curves to detect overfitting.
* **NLP:** Tokenization, Stemming, Stopword removal, and Sentiment Analysis using TF-IDF.

### 6. Capstone Project

**Lab 50:** An end-to-end Machine Learning pipeline integrating data selection, cleaning, feature engineering, model training, and evaluation into a single deployable workflow.

---

## => Comprehensive Project Guidelines & Reference Code

**How to Recreate This Project From Scratch**

These guidelines serve as the "Universal Logic" for the entire portfolio. While the repository contains 50 specific files, they all follow the **four architectural patterns** below. You can use these templates to verify or rebuild any lab in this project.

### Phase 1: Data Engineering Pattern (Universal)

**Applicable Labs:** 10–18, 35
**Concept:** Whether scraping the web or loading a CSV, the goal is to convert raw data into a clean Pandas DataFrame.
**Universal Script:**

```python
import numpy as np
import pandas as pd

# 1. Load Data (Generic approach for CSV, Excel, or SQL)
# df = pd.read_csv('data.csv') 
# For Lab 35 (Web Scraping), this data comes from BeautifulSoup parsing
data = {
    'Feature_A': [10, 20, np.nan, 40],
    'Feature_B': ['Cat', 'Dog', 'Cat', 'Bird'],
    'Target': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Universal Cleaning Logic
# Handle Missing Values (applies to Lab 14, 15)
df['Feature_A'] = df['Feature_A'].fillna(df['Feature_A'].mean())

# 3. Vectorized Math (NumPy)
# Efficiently creating new features (applies to Lab 10, 11, 33)
df['Feature_Log'] = np.log(df['Feature_A'] + 1)

print("Data Structure:\n", df.head())

```

### Phase 2: The Machine Learning Workflow (Scikit-Learn)

**Applicable Labs:** 21–42 (Regression, Classification, Ensembles)
**Concept:** This single workflow supports **every** algorithm in Scikit-Learn (Linear Regression, SVM, Decision Trees, Random Forest).
**Universal Script:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# NOTE: Import the specific algorithm you need here
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC                  # For Lab 42
from sklearn.ensemble import RandomForestClassifier # For Lab 40
from sklearn.tree import DecisionTreeClassifier     # For Lab 26

# 1. Preprocessing
# Encode categorical text to numbers (Lab 20)
le = LabelEncoder()
y = le.fit_transform(['cat', 'dog', 'cat', 'dog'])
X = [[1.2], [3.4], [1.1], [3.5]] # Dummy features

# 2. Split (Critical Step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale (Essential for SVM, k-NN, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Universal Training Logic
# CHANGE THIS LINE to use any model (SVM, Decision Tree, etc.)
model = LogisticRegression() 
# model = SVC(kernel='rbf') 
# model = RandomForestClassifier(n_estimators=100)

model.fit(X_train_scaled, y_train)

# 5. Universal Evaluation
y_pred = model.predict(X_test_scaled)
print("Model Performance:\n", classification_report(y_test, y_pred))

```

### Phase 3: Deep Learning Architecture (Keras/TensorFlow)

**Applicable Labs:** 43–46
**Concept:** The `Sequential` model is a container. The *layers* you add determine if it is an ANN, CNN, or RNN.
**Universal Script:**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, SimpleRNN

# 1. Define Architecture
model = Sequential()

# NOTE: Un-comment the block corresponding to your specific Lab

# --- Option A: Standard ANN (Lab 43) ---
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Binary Output

# --- Option B: CNN for Images (Lab 45) ---
# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax')) # Multi-class Output

# --- Option C: RNN for Sequences (Lab 46) ---
# model.add(SimpleRNN(10, input_shape=(None, 1)))
# model.add(Dense(1))

# 2. Universal Compilation
model.compile(optimizer='adam', 
              loss='binary_crossentropy', # Use 'categorical_crossentropy' for multi-class
              metrics=['accuracy'])

# 3. Training Analysis
# history = model.fit(X_train, y_train, epochs=20)

```

### Phase 4: Natural Language Processing (NLP)

**Applicable Labs:** 47–49
**Concept:** Text must be cleaned and vectorized before any Machine Learning model can read it.
**Universal Script:**

```python
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 1. Text Preprocessing Template (Lab 47)
def preprocess_text(text):
    # Standard steps: Lowercase -> Remove Punctuation -> Remove Stopwords -> Stemming
    return text.lower()

raw_data = ["AI is wonderful", "Natural Language Processing is complex"]
cleaned_data = [preprocess_text(doc) for doc in raw_data]

# 2. Vectorization (Lab 48)
# NOTE: Swap TfidfVectorizer for CountVectorizer if you only need word counts
vectorizer = TfidfVectorizer() 
X = vectorizer.fit_transform(cleaned_data)

# 3. Ready for Modeling
# This 'X' matrix can now be passed to any Phase 2 model (e.g., Logistic Regression)
print(f"Transformed Text Shape: {X.shape}")

```

---

## => Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Al-Nafi-ML-Portfolio.git](https://github.com/YourUsername/Al-Nafi-ML-Portfolio.git)

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run a specific lab:**
Navigate to the module folder and run the script:
```bash
cd 04_Machine_Learning_Models
python linear_regression.py

```



---

## => Connect

**Saleem Ali** *AI & Machine Learning Engineer*

[LinkedIn Profile](https://www.linkedin.com/in/saleem-ali) | [GitHub Repositories](https://github.com/SaleemAli?tab=repositories)

---
**Status:** Completed  
**Institution:** Al Nafi International College

```

```
