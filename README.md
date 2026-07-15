# AI-Powered Fake News Detection Using Text Classification

A machine learning pipeline built to detect and classify fake news articles using Natural Language Processing (NLP) and text classification techniques.

## Project Overview

This project was developed as part of a Summer Internship Program in AI & Machine Learning. It implements an end-to-end machine learning pipeline from scratch to distinguish between real and fake news articles. 

The pipeline includes:
- **Data Preprocessing:** Cleaning text, removing punctuation and stopwords, and tokenization.
- **Feature Engineering:** Using `TfidfVectorizer` (TF-IDF) to convert raw text into numerical features.
- **Model Building:** Training and comparing various machine learning algorithms:
  - K-Nearest Neighbors (KNN) - Non-Parametric
  - Logistic Regression - Parametric
  - Random Forest - Ensemble
  - Simple Neural Network (MLPClassifier) - Deep Learning
- **Evaluation:** Assessing models using accuracy, precision, recall, F1-score, and confusion matrices.

## Repository Contents

- `Fake_News_Detection.ipynb`: The main Jupyter Notebook containing the data pipeline, model training, and evaluation.
- `Final_Report.docx`: The comprehensive project report (IEEE format).
- `Fake_News_Detection_Presentation.pptx`: The presentation slides summarizing the project findings.
- `project_work_1.pdf`: The original project prompt and guidelines.

*(Note: The datasets `True.csv` and `Fake.csv` are not included in this repository due to size constraints. You can download the Fake News Detection Dataset from Kaggle or the UCI Repository and place the CSV files in the root directory to run the notebook.)*

## How to Run

1. Clone the repository.
2. Download the Kaggle Fake News dataset and place `Fake.csv` and `True.csv` in the root folder.
3. Open `Fake_News_Detection.ipynb` in Jupyter Notebook or Jupyter Lab.
4. Run all cells to execute the preprocessing, train the models, and view the evaluation metrics.

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-Learn
- NLTK / Regular Expressions
- Matplotlib & Seaborn (for visualizations)
