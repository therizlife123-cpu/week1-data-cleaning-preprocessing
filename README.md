# Week 1 — Data Cleaning and Preprocessing

## Project Overview
This project demonstrates a complete data acquisition, exploration, cleaning, and preprocessing workflow using the **UCI Adult Census Income Dataset**.

The work focuses on preparing a reliable dataset for subsequent statistical analysis and machine learning. The project documents missing-value handling, consistency checks, duplicate inspection, outlier analysis, categorical encoding, numerical scaling, and final validation.

## Dataset
- **Dataset:** Adult (Census Income)
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/2/adult
- **Task:** Prepare the dataset for further analysis and modelling.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- ucimlrepo

## Project Structure
```text
week1-data-cleaning-preprocessing/
├── README.md
├── week1_data_cleaning.py
├── requirements.txt
└── Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx
```

## Main Steps
1. Acquire the public dataset from UCI.
2. Inspect dataset structure, data types, and distributions.
3. Standardize column names and text values.
4. Convert source missing markers such as `?` to proper missing values.
5. Investigate missing values and duplicates.
6. Perform domain-specific validity checks.
7. Handle missing categorical and numerical values.
8. Detect numerical outliers using the IQR method.
9. Normalize the target variable.
10. Prepare categorical encoding and numerical scaling using Scikit-learn.
11. Run final validation checks.

## Key Cleaning Decisions
Missing categorical values are represented using an explicit `Unknown` category where appropriate. Numerical missing values are treated with median imputation when required. Statistical outliers are flagged for investigation rather than automatically deleted, because unusual observations may still be legitimate records.

## Report
The detailed DOCX report contains the methodology, Python code snippets, explanations, challenges, preprocessing rationale, impact on future analysis, and a complete reproducible script.

## Reproducibility
Install dependencies with:
```bash
pip install -r requirements.txt
```

Then run:
```bash
python week1_data_cleaning.py
```

## Author
Rizwan Saifi
