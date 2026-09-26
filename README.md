# Data Analytics Internship Projects — Week 1, Week 2, Week 3 & Week 4

This repository contains my internship assignments focused on **data cleaning, preprocessing, exploratory data analysis, visualization, unsupervised learning, and supervised machine learning using Python**.

## 📌 Week 1 — Data Acquisition, Cleaning & Preprocessing

### Objective
Acquire a reliable public dataset, investigate its quality, clean inconsistencies, handle missing values and outliers, and prepare the data for further analysis and machine learning.

### Dataset
- **Dataset:** UCI Adult Census Income Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/2/adult

### Work completed
- Dataset acquisition and initial inspection
- Missing-value identification and treatment
- Handling of `?` missing-value markers
- Duplicate and consistency checks
- Domain-validity checks
- IQR-based outlier analysis
- Categorical encoding strategy
- Numerical scaling strategy
- Final data-quality validation
- Discussion of preprocessing impact and data leakage

### Week 1 Files
- `week1_data_cleaning.py`
- `requirements.txt`
- `Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx`

---

## 📊 Week 2 — Exploratory Data Analysis & Visualization

### Objective
Explore a public dataset and use Python visualizations to identify trends, relationships, distributions, patterns, and potential anomalies.

### Dataset
- **Dataset:** Iris Dataset
- **Original source:** UCI Machine Learning Repository
- **Reference:** Fisher, R. A. (1936), *The use of multiple measurements in taxonomic problems*
- Distributed through `scikit-learn` for reproducibility.

### Work completed
- Dataset structure and quality inspection
- Descriptive statistics
- Species/class distribution analysis
- Grouped mean calculations
- Scatter plots, bar charts, box plots, histograms
- Correlation analysis
- IQR-based anomaly screening
- Interpretation of patterns and relationships

### Week 2 Files
- `week2_eda_visualization.py`
- `week2_requirements.txt`
- `Week_2_Exploratory_Data_Analysis_and_Visualization_Report.docx`

---

## 🤖 Week 3 — Unsupervised Learning & Clustering Analysis

### Objective
Apply unsupervised machine learning to discover meaningful groups and interpret their characteristics.

### Methodology
- Standardized numerical features using `StandardScaler`
- Evaluated k = 2 through 8
- Used the Elbow Method and Silhouette Score
- Applied K-Means with k = 3
- Visualized clusters using PCA
- Profiled cluster characteristics
- Used known species labels only for post-hoc interpretation

### Week 3 Files
- `week3_clustering_analysis.py`
- `week3_requirements.txt`
- `Week_3_Unsupervised_Learning_and_Clustering_Analysis_Report.docx`

---

## 🎯 Week 4 — Supervised Learning Model Implementation

### Objective
Build and evaluate a supervised classification model that predicts Iris species from flower measurements.

### Problem Definition
This is a **three-class classification problem**. The predictors are sepal length, sepal width, petal length, and petal width. The target is the Iris species.

### Models
- **Logistic Regression** — primary interpretable baseline
- **Random Forest** — nonlinear comparison model

### Methodology
- Created an 80/20 stratified train/test split
- Applied `StandardScaler` to Logistic Regression through a pipeline
- Used 5-fold stratified cross-validation
- Evaluated accuracy, precision, recall, and F1-score
- Generated confusion matrices
- Compared model performance
- Analyzed Random Forest feature importance
- Discussed strengths, limitations, and possible improvements

### Key Week 4 Findings
- Both models provide strong classification performance on the Iris dataset.
- Logistic Regression provides a simple and interpretable baseline.
- Random Forest provides a flexible nonlinear alternative and feature-importance estimates.
- Petal-related measurements are particularly useful for distinguishing the species.
- Cross-validation was used to check that model performance was not dependent on a single train/test split.
- Scaling was kept inside the Logistic Regression pipeline to reduce preprocessing leakage.

### Week 4 Files
- `week4_supervised_learning.py` — reproducible classification workflow
- `week4_requirements.txt` — Week 4 dependencies
- `Week_4_Supervised_Learning_Model_Implementation_Report.docx` — detailed supervised-learning report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📁 Repository Structure

```text
week1-data-cleaning-preprocessing/
│
├── README.md
├── week1_data_cleaning.py
├── requirements.txt
├── Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx
│
├── week2_eda_visualization.py
├── week2_requirements.txt
├── Week_2_Exploratory_Data_Analysis_and_Visualization_Report.docx
│
├── week3_clustering_analysis.py
├── week3_requirements.txt
├── Week_3_Unsupervised_Learning_and_Clustering_Analysis_Report.docx
│
├── week4_supervised_learning.py
├── week4_requirements.txt
└── Week_4_Supervised_Learning_Model_Implementation_Report.docx
```

## ▶️ How to Run the Projects

### Week 1
```bash
pip install -r requirements.txt
python week1_data_cleaning.py
```

### Week 2
```bash
pip install -r week2_requirements.txt
python week2_eda_visualization.py
```

### Week 3
```bash
pip install -r week3_requirements.txt
python week3_clustering_analysis.py
```

### Week 4
```bash
pip install -r week4_requirements.txt
python week4_supervised_learning.py
```

## 📚 Project Purpose

These projects demonstrate practical data-analytics and machine-learning skills, including data acquisition, data quality assessment, preprocessing, exploratory analysis, visualization, clustering, supervised classification, model validation, evaluation, interpretation, and reproducible Python workflows.

## 👤 Author

**Rizwan Saifi**

Data Analytics Internship — Week 1, Week 2, Week 3 & Week 4
