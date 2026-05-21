# Automated Cyberbullying and Toxicity Detection in Social Media

## 1. Project Title

**Automated Cyberbullying and Toxicity Detection in Social Media**

This project focuses on detecting toxic, abusive, and harmful comments in social media using Natural Language Processing and Deep Learning methods.

---

## 2. Problem Statement

### What problem are we trying to solve?

The main goal of this project is to build an automated content moderation system that can detect toxic, abusive, offensive, or harassing comments in online forums and social media platforms.

Social media platforms receive a large number of comments every day. Some of these comments may include cyberbullying, insults, threats, hate speech, or other harmful language. Manually checking every comment is difficult and time-consuming, so an automated model can help identify toxic content faster.

### Why is this problem important?

Cyberbullying is a serious problem because it can negatively affect users’ mental health and make online communities unsafe. Manual moderation is not enough for large platforms because millions of posts and comments are created daily.

An automated Deep Learning approach can provide a fast and scalable solution. It can help moderators detect harmful comments earlier and protect users from online abuse.

### What will the model predict?

Given a text comment or post, the model will predict the probability that the text contains toxic or abusive language.

This is a **multi-label classification task**, because one comment can belong to more than one category at the same time.

For example, one comment can be both:

```text
toxic + insult
```

or

```text
toxic + obscene + threat
```

---

## 3. Dataset

### Dataset Name

**Jigsaw Toxic Comment Classification Challenge**

### Dataset Source

Kaggle:

```text
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
```

In this project, a modified version of the Jigsaw dataset was used from Hugging Face:

```text
m375zhan/modified_jigsaw_toxic_comment_classification_challenge
```

### Dataset Information

The dataset contains approximately:

```text
159,000 text comments
```

Each row contains a user-generated comment and its toxicity labels.

### Input Feature

The input feature is raw text:

```text
User comment / post
```

### Target Labels

The dataset has 6 binary target labels:

| Label | Description |
|---|---|
| toxic | General toxic comment |
| severe_toxic | Very harmful toxic comment |
| obscene | Offensive or obscene language |
| threat | Threatening comment |
| insult | Insulting comment |
| identity_hate | Hate speech related to identity |

### Data Format

The dataset is stored in CSV format.

```text
CSV files
```

---

## 4. Planned Method

The project uses both a traditional machine learning baseline and a deep learning model.

### Baseline Model

The baseline model is:

```text
TF-IDF + Naive Bayes
```

TF-IDF is used to convert text into numerical features. Naive Bayes is used as a simple and fast classification algorithm for text classification.

This model was used as a baseline to compare with the deep learning model.

### Deep Learning Model

The deep learning model is based on an LSTM architecture.

```text
Embedding Layer
↓
LSTM Layer
↓
Linear Layer
↓
Sigmoid Output
```

LSTM is useful for text classification because it can learn sequential patterns in sentences and understand word order better than simple models.

### Loss Function

The loss function used in this project is:

```text
Binary Cross-Entropy Loss
```

This loss function is suitable for multi-label classification because each label is predicted independently.

### Evaluation Metrics

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures overall classification quality |
| Precision | Shows how many predicted toxic comments were actually toxic |
| Recall | Shows how many real toxic comments were correctly detected |
| F1-score | Balance between precision and recall |
| Accuracy | Overall correctness, but less reliable for imbalanced data |

Recall is very important in this project because we want to reduce false negatives. A false negative means that a toxic comment was missed by the model.

### Train / Validation / Test Split

The dataset was divided into:

```text
80% Training
10% Validation
10% Testing
```

---

## 5. Expected Challenges

### 1. Class Imbalance

Most comments in the dataset are non-toxic. Toxic labels such as `threat`, `identity_hate`, and `severe_toxic` have fewer examples.

Because of this, the model may predict common labels better than rare labels.

### 2. Context and Sarcasm

Some comments may be sarcastic or culturally specific. The model may not always understand sarcasm, slang, or hidden meaning.

### 3. Out-of-Vocabulary Words

Social media users often use misspellings, slang, abbreviations, or new words. These words may not exist in the model vocabulary.

### 4. Multi-label Prediction

One comment can belong to several toxic categories at the same time. This makes the task more difficult than simple binary classification.

---

## 6. Weekly Plan

| Week | Planned Work | Expected Output |
|---|---|---|
| Week 1 | Finalize dataset, set up GitHub repository, perform Exploratory Data Analysis on text length and label distribution | Proposal, initial README, and EDA notebook |
| Week 2 | Text preprocessing, data splitting, TF-IDF vectorization, and Naive Bayes baseline model | Baseline ROC-AUC, Precision, Recall, F1-score results |
| Week 3 | Implement LSTM model in PyTorch, train the model, and analyze training loss | Trained LSTM model, loss curves, Week 3 report |
| Week 4 | Model comparison, threshold tuning, error analysis, final documentation, and application development | Final report, final codebase, Streamlit application, presentation slides |

---

## 7. Final Project Result

By the end of the project, we created a complete toxic comment classification system.

The project includes:

```text
Dataset analysis
Text preprocessing
Baseline model
LSTM deep learning model
Model evaluation
Threshold tuning
Error analysis
Streamlit application
Final report
Presentation
```

The final application allows the user to enter a comment and receive toxicity predictions with probability scores for each category.

---

## 8. Application Description

A simple Streamlit application was created to demonstrate the model.

The application can:

```text
Take user input
Clean the text
Run the trained model
Predict toxicity labels
Show probability scores
Display whether the comment is toxic or non-toxic
```

This makes the project more practical because the model can be tested through a simple web interface.

---

## 9. Conclusion

This project demonstrates how machine learning and deep learning can be used to detect cyberbullying and toxic behavior in social media comments.

The baseline model gave strong initial results, while the LSTM model helped learn deeper text patterns. However, class imbalance remained one of the biggest challenges, especially for rare labels such as `threat`, `identity_hate`, and `severe_toxic`.

In the future, the project can be improved by using:

```text
BERT or other transformer models
Better class imbalance handling
More training epochs
Separate threshold tuning for each label
More real-world social media data
```

Overall, the project shows a working prototype of an automated toxic comment detection system that can support online moderation.
