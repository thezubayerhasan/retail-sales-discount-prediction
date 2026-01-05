# 🛒 Retail Sales Discount Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Data%20Quality%20Issues-red)

A machine learning project exploring discount prediction in retail transactions using supervised and unsupervised learning techniques.

## 📑 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation & Usage](#installation--usage)
- [Technologies](#technologies)

---

## 🎯 Overview

This project analyzes retail transaction data to predict discount application using multiple machine learning algorithms. The dataset includes transaction attributes such as product category, price, quantity, payment method, and temporal features.

**Models Implemented:**
- Neural Network (MLP Classifier)
- Decision Tree Classifier
- Naive Bayes (Gaussian)
- K-Means Clustering (Unsupervised)

---

## 📊 Dataset

**Source:** `retail_store_sales.csv`  
**Size:** 12,575 transactions (8,370 after cleaning)  
**Target:** `Discount Applied` (Binary)

**Features:**
- **Numerical:** Price Per Unit, Quantity, Total Spent
- **Categorical:** Product Category, Payment Method, Location
- **Temporal:** Transaction Date (engineered: Year, Month, Day, DayOfWeek, Quarter)
- **Final Feature Count:** 11 features after preprocessing

---

## 🔧 Data Pre-processing & Feature Engineering

### Problem 1: Missing Values ✅
**Solution 1:** Removed 4,205 rows (33.4%) with missing target variable
- Records before: 12,575
- Records after: 8,370
- Strategy: Cannot impute binary target; removal was necessary

**Solution 2:** Imputed numerical features using **median strategy**
- Features: Price Per Unit, Quantity, Total Spent
- Imputation method: Median (robust to outliers)
- ResuMethodology

### Data Preprocessing
1. **Missing Values:** Removed 4,205 rows (33.4%) with missing target variable
2. **Numerical Imputation:** Median strategy for Price, Quantity, Total Spent
3. **Encoding:** Label encoding for categorical variables
4. **Scaling:** StandardScaler normalization for all features
5. **Feature Engineering:** Extracted temporal features (Year, Month, Day, DayOfWeek, Quarter)

### Train-Test Split
- **Train:** 6,696 samples (80%)
- **Test:** 1,674 samples (20%)
- **Strategy:** Stratified sampling
**Architecture:**
- Hidden Layers: (128, 64, 32)
- Activation: ReLU
- Solver: Adam optimizer
- Max Iterations: 500
- Early Stopping: Enabled (validation_fraction=0.1)

**Performance:**
- ❌ **Accuracy:** ~51.3%
- ❌ **AUC-ROC:** ~0.51
- ⚠️ Model performed at random chance level
- ⏱️ Training time: Moderate (with early stopping but poor results)

---

### 2️⃣ Decision Tree Classifier

> **Why Decision Tree?** I implemented the Decision Tree Classifier as my second model because it provides an excellent balance between performance and interpretability. Unlike the neural network's "black box" nature, decision trees create a clear, hierarchical set of if-then rules that can be easily understood and explained to business stakeholders. The algorithm recursively splits the data based on feature values that best separate the discount and non-discount classes, creating a tree-like structure of decisions.

> **Why These Hyperparameters?** I set `max_depth=15` to limit how deep the tree can grow, which prevents it from memorizing the training data (overfitting). I also set `min_samples_split=10`, meaning a node must have at least 10 samples before it can be split further. These constraints help the model generalize better to unseen data while maintaining its ability to capture important patterns. This controlled complexity is crucial for real-world deployment where we need reliable predictions on new transactions.

**Hyperparameters:**
- Max Depth: 15
- Min Samples Split: 10

**Performance:**
- ❌ **Accuracy:** ~49.2%
- ❌ **AUC-ROC:** ~0.49
- ⚠️ Interpretable rules but no predictive power
- ❌ Random s Implemented

### 1. Neural Network (MLP Classifier)
**Architecture:** 3 hidden layers (128, 64, 32 neurons)  
**Activation:** ReLU | **Optimizer:** Adam | **Early Stopping:** Enabled

### 2. Decision Tree Classifier
**Hyperparameters:** max_depth=15, min_samples_split=10  
**Interpretable:** Rule-based decision making

### 3. Naive Bayes (Gaussian)
**Type:** GaussianNB for continuous features  
**Fast:** Quick training and prediction

### 4. K-Means Clustering
**Unsupervised:** Pattern discovery with K=3 clusters  
**Dimensionality Reduction:** PCA for visualization
### 🎨 Visualizations Generated:
- ✅ Target variable distribution (bar & pie charts)
- ✅ Numerical feature distributions (histograms)
- ✅ Discount by category, payment method, location
- ✅ Price and quantity impact on discount
- ✅ Correlation heatmap
- ✅ Model comparison charts (5 metrics)
- ✅ Comprehensive metric comparison (grouped bar chart)
- ✅ ROC curves for all models
- ✅ Confusion matrices for all models

---

## 🔍 Unsupervised Learning

### K-Means Clustering Analysis

> **Why K-Means Clustering?** After building supervised models to predict discounts, I wanted to explore the data from a different angle using unsupervised learning. K-Means clustering helps us discover natural groupings in the customer transaction data without using the discount label. This is valuable because it can reveal hidden patterns and customer segments that we might not have considered. For example, it might identify groups like "high-spending bulk buyers," "occasional shoppers," or "discount seekers" based purely on their transaction characteristics. These insights can help businesses tailor their marketing strategies and understand customer behavior beyond just whether they received a discount.

> **Why the Elbow Method?** Since K-Means requires us to specify the number of clusters beforehand, I needed a systematic way to find the optimal K. I tested K values from 2 to 10 and evaluated two metrics: **Inertia** (how tightly clustered the data points are) and **Silhouette Score** (how well-separated the clusters are). By plotting these metrics, I looked for the "elbow point" where adding more clusters doesn't significantly improve the clustering quality. This prevents us from creating too few clusters (losing important patterns) or too many clusters (overfitting and finding meaningless groups).

> **Why K=3?** Based on the Elbow Method analysis, K=3 emerged as the optimal number of clusters. The silhouette score was maximized around this point, indicating good cluster separation. This makes business sense too: we found three distinct transaction patterns in the retail data, which could represent different customer segments or purchasing behaviors. Having 3 clusters strikes a balance between simplicity (easy to understand and act upon) and detail (capturing meaningful differences in transaction patterns).

> **Why PCA for Visualization?** Our dataset has 11 features, making it impossible to visualize directly. I used **Principal Component Analysis (PCA)** to reduce these 11 dimensions down to 2 principal components while preserving 60-70% of the variance. This allows us to create a 2D scatter plot where we can actually see the three clusters and how well-separated they are. Think of PCA as finding the best "camera angles" to view the data - it identifies the two directions that capture the most variation in the transaction patterns.

**Objective:** Discover natural customer segments based on transaction patterns

#### 🔎 Elbow Method:
- Tested K = 2 to 10
- Evaluated Inertia and Silhouette Score
- **Optimal K = 3** clusters identified

#### 📊 Clustering Results:
**Final Configuration:**
- **Number of Clusters:** 3
- **Silhouette Score:** ~0.12 (very poor separation - indicates weak cluster structure)
- **Inertia:** 57,335.56

**Cluster Distribution:**
- Cluster 0: 

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Neural Network** | ~51% | ~51% | ~65% | ~57% | ~0.51 |
| **Decision Tree** | ~49% | ~49% | ~45% | ~47% | ~0.49 |
| **Naive Bayes** | ~49% | ~50% | ~54% | ~52% | ~0.49 |

### Clustering Results
- **Optimal Clusters:** K=3 (determined via Elbow Method)
- **Silhouette Score:** ~0.12 (poor separation)
- **Distribution:** 37.8%, 37.7%, 24.6%
- **Visualization:** PCA with 2 components (~35.6% variance explained)

5. **Alternative Approaches:**
   - Focus on descriptive analytics rather than prediction
   - Analyze discount patterns in subsets with clean labels
   - Consider rule-based systems if ML proves infeasible
   - Consult with business stakeholders about discount policies

---

## 👨‍💻 Contributors

**Developed by:** Md Zubayer Hasan 
**Focus Areas:** Machine Learning, Data Science, Retail Analytics  

---

## 🛠️ Tools and Libraries

### Core Libraries:
- **Python 3.9+** 🐍
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Visualization:
- **Matplotlib** - Static plots and charts
- **Seaborn** - Statistical visualizations

### Machine Learning:
- **Scikit-learn** - ML algorithms and tools
  - Classification models
  - Clustering algorithms
  - Preprocessing utilities
  - Model evaluation metrics
  - Cross-validation

### Specific Algorithms:
- **MLPClassifier** - Neural Network
- **RandomForestClassifier** - Ensemble method
- **DecisionTreeClassifier** - Tree-based model
- **KNeighborsClassifier** - Distance-based
- **LogisticRegression** - Linear model
- **GaussianNB** - Probabilistic model
- **KMeans** - Clustering
- **PCA** - Dimensionality reduction

### Development Environment:
- **Jupyter Notebook** - Interactive development
- **VS Code** - Code editing

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with proper attribution.

---

##  How to Run

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd "Is it final"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook Retail_Sales_Discount_Prediction_Zubayer.ipynb
   ```

4. **Run all cells** to reproduce the analysis and results.

---

## 📊 Project Statistics

- **Dataset Records:** 8,376 (after cleaning)
- **Features Used:** 11
- **Models Trained:** 3 supervised + 1 unsupervised
- **Training Time:** ~1-2 minutes (all models)
- **Best Model Accuracy:** ~51% (Neural Network) - **Random Chance Level**
- **Visualizations Created:** 15+
- **Project Outcome:** ⚠️ **Failed due to data quality issues**

---

⚠️ **This project demonstrates important data science lessons: recognizing data quality issues and understanding when prediction is not feasible.** ⚠️

---

*Last Updated: January 2026*  
*Project Status: ⚠️ Completed with Critical Findings - Models Unusable Due to Data Quality Issues*
�️ Technologies

**Language:** Python 3.9+

**Libraries:**
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
  - MLPClassifier (Neural Network)
  - DecisionTreeClassifier
  - GaussianNB
  - KMeans, PCA

**Environment:** Jupyter Notebook

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/retail-discount-prediction.git
   cd retail-discount-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook
   ```

4. Open `1_22201826_22299219.ipynb` and run all cells to reproduce the analysis.

---

## 📁 Project Structure

```
├── Retail_Sales_Discount_Prediction_Zubayer.ipynb     # Main analysis notebook
├── retail_store_sales.csv                              # Dataset
└── README.md                                           # Documentation
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Zubayer**

Feel free to reach out for questions or collaboration opportunities!

---

*This project demonstrates the importance of data quality in machine learning and the value of recognizing when problems are not solvable with current data.