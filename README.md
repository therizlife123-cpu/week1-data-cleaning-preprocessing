# Data Analytics Internship Projects — Week 1 & Week 2

This repository contains my first two internship assignments focused on **data cleaning, preprocessing, exploratory data analysis (EDA), and data visualization using Python**.

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
- Sepal scatter-plot analysis
- Petal scatter-plot analysis
- Grouped bar-chart comparison
- Box-plot distribution analysis
- Correlation analysis
- Petal-length distribution analysis
- IQR-based anomaly screening
- Critical interpretation of patterns and relationships
- Discussion of limitations and implications for future modelling

### Key Week 2 Findings
- The dataset contains **150 observations** across **3 balanced species classes**.
- There are **no missing values** in the Iris dataset.
- Petal length and petal width provide strong visual separation between species.
- Setosa is the most clearly separated class.
- Versicolor and virginica show more overlap than setosa.
- Petal length and petal width have a strong positive relationship.
- Sepal width has weaker relationships with the other numerical measurements.

### Week 2 Files
- `week2_eda_visualization.py` — reproducible EDA and visualization script
- `week2_requirements.txt` — Week 2 Python dependencies
- `Week_2_Exploratory_Data_Analysis_and_Visualization_Report.docx` — detailed EDA report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- ucimlrepo (Week 1)

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
└── Week_2_Exploratory_Data_Analysis_and_Visualization_Report.docx
```

## ▶️ How to Run the Projects

### Week 1

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python week1_data_cleaning.py
```

### Week 2

Install the Week 2 dependencies:

```bash
pip install -r week2_requirements.txt
```

Run the EDA script:

```bash
python week2_eda_visualization.py
```

## 📚 Project Purpose

These projects demonstrate practical data-analytics skills including data acquisition, data quality assessment, preprocessing, descriptive analysis, visualization, interpretation, and reproducible Python workflows.

## 👤 Author

**Rizwan Saifi**

Data Analytics Internship — Week 1 & Week 2
