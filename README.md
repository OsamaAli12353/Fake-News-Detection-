# Fake News Detection

## Project Description
This project focuses on detecting fake news using natural language processing (NLP) techniques and machine learning models. The goal is to classify news articles as either **FAKE** or **REAL** based on their content.

## Key Features
- Preprocessing of text data using NLP techniques such as **tokenization**, **lemmatization**, and **stopwords removal**.
- Implementation of multiple machine learning models for fake news classification, including **Transformer-based models**, **LSTM**, and **RNN**.
- Achieved high classification accuracy using **Transformer-based models** due to their ability to capture contextual information in text.

## Project Workflow
1. **Data Preprocessing**: 
   - Text cleaning: Tokenization, Lemmatization, and Stopwords removal.
   - TF-IDF vectorization to convert text into numerical features.

2. **Model Training and Evaluation**:
   - Split dataset into training and validation sets.
   - Applied various machine learning models, including Naive Bayes, Logistic Regression, and neural network-based models.
   - Evaluated model performance using accuracy metrics.

## Technologies Used
- **Python**: Programming language used for model development.
- **Libraries**: 
  - `pandas`, `numpy` for data manipulation.
  - `nltk`, `textblob` for text processing.
  - `scikit-learn` for machine learning models and feature extraction.
  - `seaborn`, `matplotlib` for data visualization.

## Model Comparison
- **Naive Bayes**: Used as a baseline classifier, achieving reasonable accuracy for the task.
- **Logistic Regression**: Applied for comparison with other models.
- **Transformer-based models**: Achieved the highest accuracy, benefiting from their ability to process sequential text data and understand context.
- **LSTM and RNN models**: Implemented as alternatives to capture the sequential nature of the text but were outperformed by Transformer models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
