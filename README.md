# 🧠 Resume Classifier 

An end-to-end machine learning pipeline to classify resumes from unstructured `.pdf`/`.docx` files and structured `.csv` metadata. Built for Piramal Finance Hackathon 2024 using advanced NLP techniques and classical ML models.

---

## 🚀 Pipeline Overview

1. **Document Parsing**  
   - `PlumPDF`, `PyPDF2`, `python-docx` used to extract raw text.
2. **Preprocessing**  
   - `regex`, `NLTK`: tokenization, stopword removal, text normalization.
3. **Feature Engineering**  
   - `TF-IDF`, `Word2Vec`, and optional LLM embeddings (e.g. OpenAI/BERT).  
   - Metadata features from CSV encoded via `LabelEncoder`.
4. **Modeling**  
   - Trained `Logistic Regression`, `Random Forest`, and `XGBoost` using `scikit-learn` and `xgboost`.
5. **Scoring & Evaluation**  
   - Accuracy, F1, ROC-AUC metrics.  
   - Resume scoring based on skill/keyword match.

---

## 🧾 Technologies Used

| Component          | Stack                                        |
|--------------------|----------------------------------------------|
| Parsing            | PlumPDF, PyPDF2, python-docx                 |
| NLP Preprocessing  | NLTK, regex, pandas, NumPy                   |
| Feature Extraction | TF-IDF, Word2Vec, LLM embeddings (optional)  |
| Modeling           | scikit-learn, XGBoost                        |
| Evaluation         | sklearn.metrics, seaborn, matplotlib         |

---
👨‍💻 Author
Built by Raghav for Piramal Finance Hackathon 2024
📫 Contact: [raghavmour8@gmail.com]


