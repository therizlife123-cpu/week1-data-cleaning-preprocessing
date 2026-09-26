# Data Analytics Internship Projects — Week 1 to Week 5

This repository contains my internship assignments focused on **data cleaning, preprocessing, exploratory data analysis, visualization, unsupervised learning, supervised machine learning, and deep learning using Python**.

## 📌 Week 1 — Data Acquisition, Cleaning & Preprocessing

### Objective
Acquire a reliable public dataset, investigate its quality, clean inconsistencies, handle missing values and outliers, and prepare the data for further analysis and machine learning.

### Dataset
- UCI Adult Census Income Dataset
- UCI Machine Learning Repository

### Week 1 Files
- `week1_data_cleaning.py`
- `requirements.txt`
- `Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx`

---

## 📊 Week 2 — Exploratory Data Analysis & Visualization

### Objective
Explore a public dataset and use Python visualizations to identify trends, relationships, distributions, patterns, and potential anomalies.

### Dataset
- Iris Dataset
- UCI-origin dataset distributed through `scikit-learn`

### Work completed
- Descriptive statistics
- Class distribution analysis
- Scatter plots and bar charts
- Box plots and histograms
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
- Standardized numerical features
- Evaluated k = 2 through 8
- Used Elbow Method and Silhouette Score
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
Build and evaluate supervised classification models that predict Iris species from flower measurements.

### Models
- Logistic Regression
- Random Forest

### Methodology
- 80/20 stratified train/test split
- Scaling through a Logistic Regression pipeline
- Five-fold stratified cross-validation
- Accuracy, precision, recall, F1-score
- Confusion matrices
- Random Forest feature importance
- Strengths, limitations, and improvement discussion

### Week 4 Files
- `week4_supervised_learning.py`
- `week4_requirements.txt`
- `Week_4_Supervised_Learning_Model_Implementation_Report.docx`

---

## 🧠 Week 5 — Deep Learning Application in Data Science

### Objective
Design, train, validate, and evaluate a neural network using a popular deep-learning framework on a public dataset.

### Problem
**Handwritten digit classification** — predict digits 0 through 9 from 8×8 grayscale image data.

### Dataset
- **Dataset:** Scikit-learn Digits Dataset
- **Observations:** 1,797
- **Input:** 64 pixel features representing 8×8 images
- **Classes:** 10 digit classes (0–9)

### Framework
- **PyTorch**

### Neural Network Architecture
```text
Input: 64 pixels
        ↓
Dense: 128 neurons + ReLU
        ↓
Dropout: 0.25
        ↓
Dense: 64 neurons + ReLU
        ↓
Dropout: 0.20
        ↓
Output: 10 classes
```

### Training Approach
- Pixel values scaled from 0–16 to approximately 0–1
- 80/20 stratified train/test split
- Additional validation split from training data
- Adam optimizer
- Learning rate: 0.001
- Cross-entropy loss
- Weight decay: 1e-4
- Dropout regularization
- Early stopping based on validation loss
- Random seed: 42

### Evaluation
- Test accuracy
- Classification metrics
- Confusion matrix
- Training/validation loss curves
- Training/validation accuracy curves
- Critical discussion of overfitting, resource constraints, limitations, and improvements

### Week 5 Files
- `week5_deep_learning.py` — PyTorch neural-network implementation
- `week5_requirements.txt` — Week 5 dependencies
- `Week_5_Deep_Learning_Application_Report.docx` — detailed deep-learning report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch

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
├── Week_4_Supervised_Learning_Model_Implementation_Report.docx
│
├── week5_deep_learning.py
├── week5_requirements.txt
└── Week_5_Deep_Learning_Application_Report.docx
```

## ▶️ How to Run

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

### Week 5
```bash
pip install -r week5_requirements.txt
python week5_deep_learning.py
```

## 📚 Project Purpose

These projects demonstrate practical data-science and machine-learning skills, including data acquisition, data quality assessment, preprocessing, exploratory analysis, visualization, clustering, supervised classification, model validation, neural-network design, deep-learning training, evaluation, interpretation, and reproducible Python workflows.

## 👤 Author

**Rizwan Saifi**

Data Analytics Internship — Week 1 to Week 5
