# Data Analytics Internship Projects — Week 1, Week 2 & Week 3

This repository contains my internship assignments focused on **data cleaning, preprocessing, exploratory data analysis, visualization, and unsupervised machine learning using Python**.

## 📌 Week 1 — Data Acquisition, Cleaning & Preprocessing

### Objective
Acquire a reliable public dataset, investigate its quality, clean inconsistencies, handle missing values and outliers, and prepare the data for further analysis and machine learning.

### Dataset
- **Dataset:** UCI Adult Census Income Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/2/adult

### Work completed
- Dataset acquisition and initial inspection
- Data type and structure analysis
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
- `week1_data_cleaning.py` — reproducible Python cleaning/preprocessing workflow
- `requirements.txt` — required Python packages
- `Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx` — detailed report

---

## 📊 Week 2 — Exploratory Data Analysis & Visualization

### Objective
Perform exploratory data analysis on a public dataset and use Python visualizations to identify trends, relationships, patterns, distributions, and potential anomalies.

### Dataset
- **Dataset:** Iris Dataset
- **Original source:** UCI Machine Learning Repository
- **Dataset reference:** Fisher, R. A. (1936), *The use of multiple measurements in taxonomic problems*
- The analysis uses the dataset distributed through `scikit-learn` for reproducibility.

### Work completed
- Dataset structure and quality inspection
- Descriptive statistics
- Species/class distribution analysis
- Grouped mean calculations
- Sepal and petal scatter plots
- Grouped bar-chart comparison
- Box-plot distribution analysis
- Correlation analysis
- Petal-length distribution analysis
- IQR-based anomaly screening
- Interpretation of patterns and relationships

### Key Week 2 Findings
- The dataset contains **150 observations** across **3 balanced species classes**.
- There are **no missing values** in the Iris dataset.
- Petal length and petal width provide strong visual separation between species.
- Setosa is the most clearly separated class.
- Versicolor and virginica show more overlap than setosa.
- Petal length and petal width have a strong positive relationship.

### Week 2 Files
- `week2_eda_visualization.py` — reproducible EDA and visualization script
- `week2_requirements.txt` — Week 2 Python dependencies
- `Week_2_Exploratory_Data_Analysis_and_Visualization_Report.docx` — detailed EDA report

---

## 🤖 Week 3 — Unsupervised Learning & Clustering Analysis

### Objective
Apply an unsupervised machine-learning technique to discover meaningful groups in a public dataset and interpret the characteristics of the resulting clusters.

### Dataset
- **Dataset:** Iris Dataset
- **Source:** UCI-origin Iris dataset distributed through `scikit-learn`
- **Observations:** 150
- **Features:** sepal length, sepal width, petal length, petal width

### Methodology
- Loaded and inspected the numerical features
- Standardized features using `StandardScaler`
- Evaluated candidate cluster counts from **k = 2 to 8**
- Used the **Elbow Method** and **Silhouette Score** to select the number of clusters
- Applied **K-Means clustering** with `k = 3`
- Visualized clusters using **PCA**
- Profiled the average characteristics of each cluster
- Performed post-hoc comparison with known species labels for interpretation only

### Key Week 3 Findings
- A three-cluster solution provides a meaningful segmentation of the Iris observations.
- Petal length and petal width are especially important in distinguishing the clusters.
- One cluster is characterized by relatively small petal measurements, while the other clusters contain progressively larger petal measurements.
- The PCA visualization shows reasonably separated groups and their centroids.
- The known species labels were **not used to train K-Means**, preserving the unsupervised-learning setup.

### Week 3 Files
- `week3_clustering_analysis.py` — reproducible K-Means clustering workflow
- `week3_requirements.txt` — Week 3 Python dependencies
- `Week_3_Unsupervised_Learning_and_Clustering_Analysis_Report.docx` — detailed clustering report

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
│
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
└── Week_3_Unsupervised_Learning_and_Clustering_Analysis_Report.docx
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

## 📚 Project Purpose

These projects demonstrate practical data-analytics and machine-learning skills, including data acquisition, data quality assessment, preprocessing, exploratory analysis, visualization, statistical interpretation, clustering, model evaluation, and reproducible Python workflows.

## 👤 Author

**Rizwan Saifi**

Data Analytics Internship — Week 1, Week 2 & Week 3
