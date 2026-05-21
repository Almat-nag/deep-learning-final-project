# Automated Detection of Cyberbullying and Toxic Behavior in Social Networks

## Project Overview

This project focuses on automated detection of cyberbullying and toxic behavior in social networks. The main goal of the project is to build a machine learning and deep learning system that can identify toxic comments and classify them into different toxicity categories.

The project was completed during Week 1–Week 4. Each week focused on a different stage of the machine learning pipeline: dataset analysis, preprocessing, baseline model, LSTM model, model comparison, error analysis, and application development.

---

## Problem Statement

Social networks contain a very large number of comments every day. Some of these comments may include toxic language, insults, threats, hate speech, or cyberbullying.

Manual moderation is difficult because it takes a lot of time and human effort. Therefore, automatic detection of toxic comments can help make online platforms safer and more comfortable for users.

This project solves this problem by using Natural Language Processing and classification models.

---

## Dataset

The dataset used in this project is based on the Jigsaw Toxic Comment Classification dataset.

Dataset used:

```
m375zhan/modified_jigsaw_toxic_comment_classification_challenge
```

The dataset contains:

```
159,571 comments
1 text column
6 target labels
```

Target labels:

| Label | Description |
|---|---|
| toxic | General toxic comment |
| severe_toxic | Very harmful toxic comment |
| obscene | Obscene or offensive language |
| threat | Threatening comment |
| insult | Insulting comment |
| identity_hate | Hate speech related to identity |

This is a multi-label classification task. It means that one comment can have more than one label at the same time.

For example, one comment can be both toxic and insulting.

---

## Technologies Used

This project was implemented using:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Hugging Face Datasets
- Streamlit
- Google Colab
- GitHub

---

# Week 1: Dataset Selection and Exploratory Data Analysis

## Goal of Week 1

The main goal of Week 1 was to choose a dataset, load it, understand its structure, and analyze the distribution of toxic comment labels.

## Work Completed

During Week 1, the following tasks were completed:

- Loaded the dataset
- Checked dataset shape
- Checked column names
- Checked missing values
- Analyzed target labels
- Counted how many toxic comments exist in each category
- Visualized label distribution
- Analyzed text length

## Dataset Shape

```
159571 rows, 8 columns
```

## Missing Values

```
text             0
toxic            0
severe_toxic     0
obscene          0
threat           0
insult           0
identity_hate    0
```

There were no missing values in the main text and label columns.

## Label Distribution

```
toxic            15294
obscene           8449
insult            7877
severe_toxic      1595
identity_hate     1405
threat             478
```

## Week 1 Result

From Week 1 analysis, we found that the dataset is highly imbalanced. Some classes such as `toxic`, `obscene`, and `insult` appear more often, while `threat`, `identity_hate`, and `severe_toxic` appear much less frequently.

This is important because imbalanced data can make model training more difficult. The model may learn common labels better than rare labels.

---

# Week 2: Text Preprocessing and Baseline Model

## Goal of Week 2

The goal of Week 2 was to clean the text data and train a simple baseline machine learning model.

## Work Completed

During Week 2, the following tasks were completed:

- Cleaned text data
- Converted text to lowercase
- Removed links
- Removed special characters
- Removed extra spaces
- Created cleaned text column
- Split data into train, validation, and test sets
- Applied TF-IDF vectorization
- Trained a Naive Bayes baseline model
- Evaluated the model

## Text Preprocessing

Text preprocessing is important because raw comments may contain unnecessary symbols, links, capital letters, and extra spaces. Cleaning the text helps the model understand the data better.

Example:

```
Original text:
Explanation why the edits made under my username...

Cleaned text:
explanation why the edits made under my username...
```

## Data Split

```
Train size: 127656
Validation size: 15957
Test size: 15958
```

## Baseline Model

The baseline model used in Week 2 was:

```
TF-IDF + Multinomial Naive Bayes
```

TF-IDF was used to convert text into numerical features. Naive Bayes was used as a simple and fast classification algorithm.

## TF-IDF Feature Shape

```
Train: (127656, 50000)
Validation: (15957, 50000)
Test: (15958, 50000)
```

## Week 2 Validation Results

| Label | ROC-AUC | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|---:|
| toxic | 0.9441 | 0.9513 | 0.4193 | 0.5820 | 0.9420 |
| severe_toxic | 0.9401 | 0.2857 | 0.0126 | 0.0241 | 0.9898 |
| obscene | 0.9450 | 0.9577 | 0.3690 | 0.5328 | 0.9652 |
| threat | 0.8438 | 0.0000 | 0.0000 | 0.0000 | 0.9977 |
| insult | 0.9350 | 0.8649 | 0.2779 | 0.4207 | 0.9613 |
| identity_hate | 0.8801 | 1.0000 | 0.0064 | 0.0127 | 0.9903 |

## Average Week 2 Results

```
ROC-AUC:   0.9147
Precision: 0.6766
Recall:    0.1809
F1-score:  0.2621
Accuracy:  0.9744
```

## Week 2 Result

The baseline model showed good ROC-AUC and accuracy. However, recall was low. This means the model was good at making confident predictions, but it missed many toxic comments.

The main reason is class imbalance. Rare categories such as `threat` and `identity_hate` were difficult for the model to detect.

---

# Week 3: LSTM Deep Learning Model

## Goal of Week 3

The goal of Week 3 was to build a deep learning model using PyTorch. Instead of using only TF-IDF features, we used an LSTM model that can learn patterns from word sequences.

## Work Completed

During Week 3, the following tasks were completed:

- Cleaned text data
- Built vocabulary
- Converted text into sequences
- Added padding to make sequences the same length
- Created PyTorch Dataset
- Created DataLoader
- Built LSTM model
- Trained the model
- Evaluated validation results
- Saved the trained model

## Model Architecture

The LSTM model architecture was:

```
Embedding Layer
↓
LSTM Layer
↓
Linear Layer
↓
Sigmoid Output
```

Because this is a multi-label classification task, the model used:

```
BCEWithLogitsLoss
```

This loss function is suitable when each comment can belong to several classes at the same time.

## Training Results

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.1638 | 0.1423 |
| 2 | 0.1447 | 0.1420 |
| 3 | 0.1442 | 0.1422 |
| 4 | 0.1391 | 0.1081 |
| 5 | 0.0964 | 0.0798 |

## Training Interpretation

Train loss and validation loss decreased during training. This means that the model was learning useful patterns from the comments.

The validation loss also decreased, which means the model was not only memorizing the training data, but also performing better on unseen validation data.

## Week 3 LSTM Validation Results

| Label | ROC-AUC | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|---:|
| toxic | 0.8837 | 0.8151 | 0.4867 | 0.6095 | 0.9390 |
| severe_toxic | 0.9746 | 0.0000 | 0.0000 | 0.0000 | 0.9896 |
| obscene | 0.9467 | 0.7745 | 0.6124 | 0.6840 | 0.9708 |
| threat | 0.8842 | 0.0000 | 0.0000 | 0.0000 | 0.9976 |
| insult | 0.9282 | 0.7377 | 0.5357 | 0.6207 | 0.9670 |
| identity_hate | 0.8893 | 0.0000 | 0.0000 | 0.0000 | 0.9908 |

## Average Week 3 Results

```
ROC-AUC:   0.9178
Precision: 0.3879
Recall:    0.2725
F1-score:  0.3190
Accuracy:  0.9758
```

## Week 3 Result

The LSTM model improved recall and F1-score compared to the baseline model. This means that the deep learning model was better at detecting toxic comments.

However, rare labels such as `threat`, `identity_hate`, and `severe_toxic` were still difficult to predict because the dataset contains very few examples of these classes.

---

# Week 4: Model Comparison, Threshold Tuning, Error Analysis, and Application

## Goal of Week 4

The goal of Week 4 was to finalize the project. In this week, we compared models, tested different thresholds, analyzed errors, saved the final model, and created a simple application.

## Work Completed

During Week 4, the following tasks were completed:

- Trained and compared models
- Tested different classification thresholds
- Selected the best threshold
- Performed error analysis
- Saved final model files
- Created a Streamlit application
- Prepared final results

## Threshold Tuning

In multi-label classification, the model returns probabilities. To convert probabilities into labels, we need a threshold.

For example:

```
If probability >= threshold, the label is predicted as 1.
If probability < threshold, the label is predicted as 0.
```

Different thresholds were tested.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.2 | 0.3952 | 0.0181 | 0.0337 |
| 0.3 | 0.5028 | 0.0181 | 0.0342 |
| 0.4 | 0.5833 | 0.0181 | 0.0342 |
| 0.5 | 0.6250 | 0.0171 | 0.0323 |

Best threshold:

```
0.4
```

## Week 4 Model Comparison

| Model | ROC-AUC | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|---:|
| Week 2 TF-IDF + Naive Bayes | 0.8815 | 0.4957 | 0.0651 | 0.1136 | 0.9677 |
| Week 4 LSTM + Threshold Tuning | 0.5667 | 0.5833 | 0.0181 | 0.0342 | 0.9632 |

## Week 4 Result

The Week 4 experiment showed that threshold tuning can change model performance. After threshold tuning, the LSTM model had higher precision, but recall and F1-score were lower.

This means that the model became more careful and predicted fewer toxic labels. As a result, it made fewer false positive predictions, but it also missed more toxic comments.

This result also shows that class imbalance is a serious challenge in this project.

---

# Streamlit Application

## Application Overview

A simple Streamlit application was created to demonstrate the final model.

The application allows users to enter a comment and get toxicity predictions.

## Application Features

The application can:

- Take a user comment as input
- Clean and preprocess the comment
- Convert the text into numerical format
- Run the trained model
- Show probability scores for each toxicity label
- Display predicted toxic categories
- Show whether the comment is toxic or non-toxic

## Example Use Case

```
User enters a social media comment.
The application analyzes the text.
The model predicts toxicity probabilities.
The application shows labels such as toxic, obscene, insult, threat, severe_toxic, or identity_hate.
```

## Why the Application Is Useful

The application makes the project more practical. Instead of only showing results in notebooks, the model can be tested through a simple user interface.

This shows how the model could be used in a real social media moderation system.

---

# Final Results Summary

| Week | Main Work | Result |
|---|---|---|
| Week 1 | Dataset loading and EDA | Dataset structure and class imbalance were analyzed |
| Week 2 | Text preprocessing and baseline model | TF-IDF + Naive Bayes achieved average ROC-AUC 0.9147 |
| Week 3 | LSTM deep learning model | LSTM improved average F1-score to 0.3190 |
| Week 4 | Threshold tuning, comparison, error analysis, app | Final model and Streamlit application were prepared |

---

# Key Findings

The dataset is imbalanced. Some toxic categories have many examples, while others have very few examples.

The baseline TF-IDF + Naive Bayes model performed well in terms of ROC-AUC and accuracy.

The LSTM model improved recall and F1-score in Week 3 because it learned sequential patterns in text.

Rare classes such as `threat`, `identity_hate`, and `severe_toxic` were difficult to predict.

Accuracy alone is not enough for this project because the dataset is imbalanced.

ROC-AUC, precision, recall, and F1-score are more useful metrics for this task.

The Streamlit application makes the project easier to test and present.

---

# Project Structure

```
DL_project/
│
├── notebooks/
│   ├── week1.ipynb
│   ├── week2.ipynb
│   ├── week3.ipynb
│   └── week4.ipynb
│
├── results/
│   ├── week3_training_history.csv
│   ├── week3_lstm_model.pt
│   ├── week4_training_history.csv
│   ├── week4_model_comparison.csv
│   ├── week4_threshold_tuning_results.csv
│   ├── week4_error_analysis.csv
│   └── final_lstm_model.pt
│
├── app.py
├── requirements.txt
└── README.md
```

---

# How to Run the Project

First, install the required libraries:

```
pip install pandas numpy matplotlib scikit-learn torch datasets streamlit
```

To run the Streamlit application:

```
streamlit run app.py
```

---

# Conclusion

This project successfully developed a toxic comment classification system using machine learning and deep learning methods.

During Week 1–Week 4, the project went through the full machine learning pipeline:

- Dataset analysis
- Text preprocessing
- Baseline model training
- LSTM model training
- Model evaluation
- Threshold tuning
- Error analysis
- Application development

The best Week 3 LSTM experiment achieved:

```
Average ROC-AUC:   0.9178
Average F1-score:  0.3190
Average Accuracy:  0.9758
```

The final result is a working prototype for detecting cyberbullying and toxic behavior in social media comments.

In the future, this project can be improved by using better class imbalance handling, more training epochs, pretrained models such as BERT, and separate thresholds for each toxicity label.
