# 📰 Fake News Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)

A machine learning project that detects whether a news article is **fake** or **real** using NLP and classification algorithms. Built to combat misinformation and promote media literacy.

---

## 📦 Dataset

**Source**: [Kaggle – Fake News Detection Datasets by Emine Yetim](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

**Files Included**:
- `True.csv`: Real news articles  
- `Fake.csv`: Fake news articles  

**Features**:
- `title`: Headline of the article  
- `text`: Full article content  
- `subject`: Topic category  
- `date`: Publication date  

---

## 🎯 Objectives

- Clean and preprocess raw text data  
- Perform exploratory data analysis (EDA)  
- Extract features using TF-IDF and CountVectorizer  
- Train multiple machine learning models  
- Evaluate performance using standard metrics  
- Save and deploy the best-performing model  

---

## 🧰 Tech Stack

| Category         | Tools Used                          |
|------------------|-------------------------------------|
| Language         | Python 3.x                          |
| Data Handling    | Pandas, NumPy                       |
| NLP              | NLTK, SpaCy                         |
| ML Models        | Scikit-learn, XGBoost               |
| Visualization    | Matplotlib, Seaborn                 |
| Deployment       | Pickle, Streamlit (optional)        |

---

## 🧪 Models Implemented

- ✅ Logistic Regression  
- ✅ Naive Bayes  
- ✅ Support Vector Machine (SVM)  
- ✅ Random Forest  
- ✅ XGBoost  

**Evaluation Metrics**:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 📊 Sample Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 97.8%    | 98.1%     | 97.5%  | 97.8%    |
| Random Forest       | 96.4%    | 96.7%     | 96.1%  | 96.4%    |
| XGBoost             | 98.2%    | 98.4%     | 98.0%  | 98.2%    |

📈 *Visualizations and confusion matrices available in the `results/` folder.*

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/KardamSinghal/Fake-News-Detection.git
cd Fake-News-Detection

### 2. Install Dependencies
pip install -r requirements.txt

### Sure thing, Kardam! Here's your complete, professional README.md file—all in one piece, ready to copy and paste directly into your GitHub repository:

# 📰 Fake News Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)

A machine learning project that detects whether a news article is **fake** or **real** using NLP and classification algorithms. Built to combat misinformation and promote media literacy.

---

## 📦 Dataset

**Source**: [Kaggle – Fake News Detection Datasets by Emine Yetim](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

**Files Included**:
- `True.csv`: Real news articles  
- `Fake.csv`: Fake news articles  

**Features**:
- `title`: Headline of the article  
- `text`: Full article content  
- `subject`: Topic category  
- `date`: Publication date  

---

## 🎯 Objectives

- Clean and preprocess raw text data  
- Perform exploratory data analysis (EDA)  
- Extract features using TF-IDF and CountVectorizer  
- Train multiple machine learning models  
- Evaluate performance using standard metrics  
- Save and deploy the best-performing model  

---

## 🧰 Tech Stack

| Category         | Tools Used                          |
|------------------|-------------------------------------|
| Language         | Python 3.x                          |
| Data Handling    | Pandas, NumPy                       |
| NLP              | NLTK, SpaCy                         |
| ML Models        | Scikit-learn, XGBoost               |
| Visualization    | Matplotlib, Seaborn                 |
| Deployment       | Pickle, Streamlit (optional)        |

---

## 🧪 Models Implemented

- ✅ Logistic Regression  
- ✅ Naive Bayes  
- ✅ Support Vector Machine (SVM)  
- ✅ Random Forest  
- ✅ XGBoost  

**Evaluation Metrics**:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 📊 Sample Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 97.8%    | 98.1%     | 97.5%  | 97.8%    |
| Random Forest       | 96.4%    | 96.7%     | 96.1%  | 96.4%    |
| XGBoost             | 98.2%    | 98.4%     | 98.0%  | 98.2%    |

📈 *Visualizations and confusion matrices available in the `results/` folder.*

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/KardamSinghal/Fake-News-Detection.git
cd Fake-News-Detection

### 2. Install Dependencies
pip install -r requirements.txt


### 3. Download Dataset
Download from Kaggle and place True.csv and Fake.csv in the data/ folder.

### 4. Run the Notebook
jupyter notebook notebooks/FakeNewsDetection.ipynb

📁 Project Structure
Fake-News-Detection/
│
├── data/                  # Raw dataset files
├── notebooks/             # Jupyter notebooks
├── models/                # Saved model files
├── results/               # Evaluation metrics and plots
├── utils/                 # Helper functions
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

🔮 Future Enhancements
- Integrate deep learning models (LSTM, BERT)
- Build a web app using Streamlit or Flask
- Add multilingual news support
- Improve model explainability with SHAP or LIME

🙌 Acknowledgments
- Kaggle Dataset by Emine Yetim
- Scikit-learn documentation
- NLTK and SpaCy for NLP tools

📬 Contact
Kardam Singhal
🔗 [LinkedIn](https://www.linkedin.com/in/kardamsinghal)  
📫 Email: kardamsinghalllll@gmail.com


📄 License
This project is licensed under the MIT License. See the LICENSE file for details
