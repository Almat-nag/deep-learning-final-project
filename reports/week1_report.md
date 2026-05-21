# Week 1 Report: Dataset Selection and Exploratory Data Analysis

## Project Title

Automated Detection of Cyberbullying and Toxic Behavior in Social Networks

## 1. Introduction

The goal of this project is to create a machine learning system that can automatically detect toxic and offensive comments in online platforms. Toxic comments may include insults, hate speech, threats, and cyberbullying behavior.

Online platforms generate a huge amount of user comments every day, and manual moderation is difficult and time-consuming. Automated toxic comment detection systems can help improve online safety and reduce harmful content.

The project uses the following dataset:

`m375zhan/modified_jigsaw_toxic_comment_classification_challenge`

---

## 2. Dataset Description

The dataset was loaded using Hugging Face Datasets.

### Dataset Information

| Item          |   Value |
| ------------- | ------: |
| Total rows    | 159,571 |
| Total columns |       8 |

### Dataset Columns

* `id`
* `text`
* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

The dataset contains text comments and binary labels for different toxic categories.

---

## 3. Exploratory Data Analysis (EDA)

### Missing Values

The dataset was checked for missing values.

Result:

* no missing values were found in the text column
* no missing values were found in target label columns

This means the dataset is clean and ready for preprocessing.

---

### Label Distribution

The frequency of toxic labels was analyzed.

Main observations:

* `toxic` is the most common label
* `threat` is the rarest label
* the dataset is highly imbalanced

This imbalance may create difficulties during model training because some classes have very few examples.

---

### Text Length Analysis

The length of comments was analyzed by counting the number of words in each text.

Main findings:

* some comments are very short
* some comments are extremely long
* average text length is moderate
* text lengths vary significantly across the dataset

A histogram was created to visualize text length distribution.

---

## 4. Expected Challenges

Possible project difficulties include:

* imbalanced label distribution
* noisy online text
* short comments with limited context
* rare toxic categories
* overfitting during deep learning training
* long training time for neural networks

---

## 5. Planned Method

The project will compare a baseline machine learning model and a deep learning model.

### Baseline Model

* TF-IDF Vectorizer
* Naive Bayes classifier

### Deep Learning Model

* LSTM neural network using PyTorch

### Evaluation Metrics

* ROC-AUC
* Precision
* Recall
* F1-score
* Accuracy

---

## 6. Weekly Plan

| Week   | Planned Work                             | Expected Output               |
| ------ | ---------------------------------------- | ----------------------------- |
| Week 1 | Dataset selection, EDA, repository setup | Proposal and dataset analysis |
| Week 2 | Text preprocessing and baseline model    | Baseline evaluation results   |
| Week 3 | Deep learning model training             | LSTM model and experiments    |
| Week 4 | Final improvements and evaluation        | Final report and presentation |

---

## 7. Conclusion

In Week 1, the dataset was selected, inspected, and analyzed using exploratory data analysis techniques.

The project repository structure was prepared, the dataset was explored, and the main project challenges were identified. This week created the foundation for preprocessing and model training in the next stages of the project.
