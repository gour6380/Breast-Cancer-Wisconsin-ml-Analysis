# Breast Cancer Wisconsin ML Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.2-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.0-blue.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project that classifies breast cancer diagnoses (malignant vs benign) using the Wisconsin Breast Cancer Diagnostic dataset. Built as part of the Udacity AI Programming with Python Nanodegree.

## 📊 Project Overview

This project implements a **Logistic Regression** model to predict breast cancer diagnoses from tumor cell measurements. The model achieves **96.49% accuracy** and **99.60% ROC AUC** on the test set, demonstrating strong predictive performance for medical classification.

### Key Highlights
- ✅ Comprehensive data preprocessing with StandardScaler
- ✅ Exploratory data analysis with feature distribution visualizations
- ✅ Well-documented Jupyter notebook with markdown explanations
- ✅ Professional evaluation with multiple metrics (accuracy, ROC AUC, precision, recall)
- ✅ Visual analysis with confusion matrix and ROC curve
- ✅ Academic-style analysis report with peer-reviewed citations

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.49% |
| **ROC AUC** | 99.60% |
| **Precision (Benign)** | 95.95% |
| **Recall (Benign)** | 98.61% |
| **Precision (Malignant)** | 97.50% |
| **Recall (Malignant)** | 92.86% |

**Confusion Matrix:**
- True Negatives: 71
- False Positives: 1
- False Negatives: 3
- True Positives: 39

## 📁 Project Structure

```
.
├── README.md                              # Project documentation
├── modeling.ipynb                         # Main Jupyter notebook with analysis
├── Machine_Learning_Analysis_Report.pdf   # Comprehensive analysis report
├── requirements.txt                       # Python dependencies
├── data/
│   └── data.csv                          # Wisconsin Breast Cancer dataset
└── scaler_stats.csv                      # Fitted scaler parameters
```

## 🗃️ Dataset

**Wisconsin Breast Cancer (Diagnostic) Dataset**

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances**: 569 patients (357 benign, 212 malignant)
- **Features**: 30 numerical features computed from digitized FNA images
- **Target**: Binary diagnosis (M = Malignant, B = Benign)

**Citation:**
```
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). 
Breast Cancer Wisconsin (Diagnostic) [Dataset]. 
UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
```

### Features
The dataset includes 10 base measurements (mean, standard error, and "worst" value for each):
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- pip package manager

### Key Dependencies
- **numpy==2.4.2** - Numerical computing
- **pandas==3.0.0** - Data manipulation and analysis
- **scikit-learn==1.8.0** - Machine learning algorithms
- **matplotlib==3.10.8** - Data visualization
- **seaborn==0.13.2** - Statistical data visualization
- **jupyter==** (via ipykernel==7.2.0) - Interactive notebook environment

See `requirements.txt` for complete list of dependencies.

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/gour6380/Breast-Cancer-Wisconsin-ml-Analysis.git
cd Breast-Cancer-Wisconsin-ml-Analysis
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

1. **Run the Jupyter notebook**
```bash
jupyter notebook modeling.ipynb
```

2. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Perform exploratory data analysis
   - Preprocess features with StandardScaler
   - Train Logistic Regression model
   - Evaluate performance with multiple metrics
   - Visualize results (confusion matrix, ROC curve)

## 🔬 Methodology

### 1. Data Preparation
- Removed empty columns (`Unnamed: 32`)
- Encoded target variable (M→1, B→0)
- Performed exploratory data analysis
- Split data: 80% training, 20% testing (stratified)

### 2. Preprocessing
- **StandardScaler**: Normalized features to mean=0, std=1
- **Pipeline approach**: Prevents data leakage by fitting scaler only on training data

### 3. Model Selection
- **Algorithm**: Logistic Regression
- **Rationale**: 
  - Appropriate for binary classification
  - Interpretable coefficients for medical context
  - Well-calibrated probability estimates
  - Proven effectiveness in medical ML tasks

### 4. Evaluation
- Multiple metrics: Accuracy, ROC AUC, Precision, Recall, F1-score
- Confusion matrix analysis
- ROC curve visualization
- Feature importance analysis

## 📈 Top Predictive Features

The model identified these features as most predictive (by absolute coefficient):

1. **smoothness_mean** (1.43)
2. **concavity_se** (1.23)
3. **texture_se** (1.06)
4. **concave points_se** (0.95)
5. **symmetry_worst** (0.91)

## 📝 Documentation

- **`modeling.ipynb`**: Fully documented notebook with markdown explanations, code, visualizations
- **`Machine_Learning_Analysis_Report.pdf`**: Comprehensive analysis report including:
  - Dataset description
  - Modeling approach and justification
  - Results interpretation
  - Clinical context
  - Limitations and bias discussion
  - References (8 sources, 4 scholarly)

## ⚠️ Limitations & Future Work

### Current Limitations
- Class imbalance (63% benign, 37% malignant)
- Fixed 0.5 classification threshold (not optimized for medical context)
- Single model architecture (no ensemble comparison)
- Limited to one institution's data (1990s Wisconsin)

### Future Improvements
- Threshold optimization to minimize false negatives
- Ensemble methods (Random Forest, XGBoost)
- Cross-validation for robust performance estimates
- External validation on diverse datasets
- Hyperparameter tuning via grid search

## 🤝 Contributing

This is an academic project for the Udacity AI Programming with Python Nanodegree. Feedback and suggestions are welcome!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Gourav Karwasara**
- GitHub: [@gour6380](https://github.com/gour6380)
- Project: Udacity AI Programming with Python Nanodegree - Machine Learning Foundations

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the dataset
- **Udacity** for the AI Programming with Python Nanodegree program
- **scikit-learn** community for excellent documentation and tools
- Original dataset creators: **Wolberg, Mangasarian, Street, Street (1993)**

## 📚 References

Key references used in this analysis:

1. Dreiseitl, S., & Ohno-Machado, L. (2002). Logistic regression and artificial neural network classification models: A methodology review. *Journal of Biomedical Informatics*.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

3. Sidey-Gibbons, J. A. M., & Sidey-Gibbons, C. J. (2019). Machine learning in medicine: A practical introduction. *BMC Medical Research Methodology*.

See `Machine_Learning_Analysis_Report.pdf` for complete bibliography.

---

**⭐ If you find this project helpful, please consider giving it a star!**
