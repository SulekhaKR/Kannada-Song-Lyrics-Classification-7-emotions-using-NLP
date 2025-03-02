# Kannada Song Emotion Classification using NLP

## Overview

This project focuses on classifying Kannada song lyrics into seven different emotions (Rasas) using Natural Language Processing (NLP) and Machine Learning (ML). The classification is achieved through text preprocessing, feature extraction, and training a supervised model.

## Objective

The primary goal of this project is to develop a robust ML model that can accurately classify Kannada song lyrics into the following seven emotions:

- **Love**
- **Sad**
- **Fear**
- **Devotional**
- **Patriotic**
- **Folk**
- **Happy**

## Dataset

- Source: The dataset is collected from **Sahitya Sudhe**.
- Contains 1000+ Kannada song lyrics.
- Preprocessed and balanced using **SMOTE-Tomek Links** to ensure fair representation of all classes.

## Tools and Libraries Used

- **Python** (Primary programming language)
- **Pandas** (Data processing and analysis)
- **Scikit-learn** (Machine Learning algorithms and evaluation)
- **IndicNLP Library** (Preprocessing Kannada text, tokenization, stopword removal, and normalization)
- **TfidfVectorizer** (Feature extraction using character n-grams)
- **Support Vector Machine (SVM)** (Text classification)
- **Imbalanced-learn (SMOTE-Tomek Links, ADASYN)** (Class balancing techniques)
- **Matplotlib & Seaborn** (Data visualization)

## Methodology

1. **Data Preprocessing**

   - Used **IndicNLP** for text normalization, tokenization, and stopword removal.
   - Removed special characters, extra spaces, and unnecessary symbols.
   - Converted text to a numerical representation using **TF-IDF (character n-grams up to 5-grams)**.

2. **Data Augmentation and Balancing**

   - Applied **SMOTE-Tomek Links** to balance class distribution, ensuring equal representation of all seven emotions.

3. **Model Training and Hyperparameter Tuning**

   - Implemented **Support Vector Machine (SVM)** with hyperparameter tuning using **RandomizedSearchCV**.
   - Optimized hyperparameters such as `C`, `gamma`, and `kernel`.

4. **Evaluation**

   - Measured model performance using **accuracy, precision, recall, F1-score, and confusion matrices**.
   - Achieved an accuracy of **90%+** after class balancing and hyperparameter tuning.

## Results

- **Accuracy Achieved**: \~90%
- **Best Performing Model**: Support Vector Machine (SVM)
- **Balanced Classification Across Emotions**

## How to Use

### 1. Install Dependencies

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn indic-nlp-library
```

### 2. Run the Model

```python
from model import predict_rasa
lyrics = "ನಮ್ಮ ದೇಶದ ಧ್ವಜ ಹಾರಲಿ, ಜಯ ಭಾರತ ಮಾತಾ!"
predicted_rasa = predict_rasa(lyrics)
print("Predicted Emotion:", predicted_rasa)
```

## Future Improvements

- Experimenting with **deep learning models (LSTMs, Transformers)** for improved accuracy.
- Expanding dataset with more songs from diverse sources.
- Fine-tuning **IndicBERT or mBERT** for better contextual understanding of Kannada lyrics.

##

## Acknowledgments

- **Sahitya Sudhe** for providing Kannada song lyrics.
- **IndicNLP team** for developing language processing tools for Indic languages.



