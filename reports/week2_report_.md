# Week 2 Report: Text Preprocessing and Baseline Model

## Project Title

Automated Detection of Cyberbullying and Toxic Behavior in Social Networks

## 1. Introduction

The goal of this project is to build a machine learning system that can automatically detect toxic and offensive comments in online platforms. Toxic comments may include insults, threats, hate speech, or cyberbullying behavior. Manual moderation of large social media platforms takes a lot of time, so automated moderation systems are important.

In Week 2, the main focus was on text preprocessing, exploratory data analysis, dataset preparation, TF-IDF feature extraction, training a baseline model, and evaluating model performance.

The project uses the dataset: `m375zhan/modified_jigsaw_toxic_comment_classification_challenge`.

## 2. Dataset Description

The dataset contains online user comments. Each comment can belong to one or more toxic categories, so this is a multi-label classification problem.

**Dataset information:**

| Item            |   Value |
| --------------- | ------: |
| Total rows      | 159,571 |
| Total columns   |       8 |
| Train size      | 127,656 |
| Validation size |  15,957 |
| Test size       |  15,958 |

**Columns used in the dataset:**

* `id`
* `text`
* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

## 3. Missing Values

The dataset was checked for missing values. There were no missing values in the text column or target label columns.

| Column        | Missing values |
| ------------- | -------------: |
| text          |              0 |
| toxic         |              0 |
| severe_toxic  |              0 |
| obscene       |              0 |
| threat        |              0 |
| insult        |              0 |
| identity_hate |              0 |

After checking missing values, the dataset shape remained the same: **159,571 rows and 8 columns**.

## 4. Text Preprocessing

Before training the model, the text data was cleaned and normalized. This step helps the model work with more consistent text.

The preprocessing steps included:

* converting text to lowercase
* removing URLs
* removing punctuation
* removing numbers
* removing extra spaces
* cleaning noisy symbols

Example:

```text
Original: YOU are stupid!!! Visit http://test.com
Cleaned: you are stupid
```

## 5. Exploratory Data Analysis

### Text Length Statistics

A new column called `text_length` was created to measure the number of words in each comment.

| Statistic          |   Value |
| ------------------ | ------: |
| Count              | 159,571 |
| Mean               |   67.65 |
| Standard deviation |  100.36 |
| Minimum            |       0 |
| 25%                |      16 |
| Median             |      36 |
| 75%                |      76 |
| Maximum            |    1403 |

The average comment length is around 68 words. However, some comments are very short, while some are much longer.

### Label Distribution

| Label         | Number of comments |
| ------------- | -----------------: |
| toxic         |             15,294 |
| obscene       |              8,449 |
| insult        |              7,877 |
| severe_toxic  |              1,595 |
| identity_hate |              1,405 |
| threat        |                478 |

The most common label is `toxic`, while the least common label is `threat`. This shows that the dataset is imbalanced. Some classes have many examples, while others have very few.

## 6. Train, Validation, and Test Split

The dataset was divided into three parts:

| Split      |    Size |
| ---------- | ------: |
| Train      | 127,656 |
| Validation |  15,957 |
| Test       |  15,958 |

The training set was used to train the model. The validation set was used to check model performance. The test set can be used later for final evaluation.

## 7. Feature Extraction Using TF-IDF

Machine learning models cannot directly understand raw text, so the comments were converted into numerical vectors using TF-IDF vectorization.

TF-IDF helps identify important words in comments. Words that are frequent in one comment but not common everywhere receive higher importance.

**TF-IDF shapes:**

| Split      |           Shape |
| ---------- | --------------: |
| Train      | (127656, 50000) |
| Validation |  (15957, 50000) |
| Test       |  (15958, 50000) |

The vectorizer created **50,000 features** for each dataset split.

## 8. Baseline Model

The baseline model was created using:

* TF-IDF Vectorizer
* Multinomial Naive Bayes
* MultiOutputClassifier

This model was selected because it is simple, fast, and commonly used for text classification tasks. It provides a good baseline for comparison with future deep learning models.

The model produced predictions with the following shapes:

| Output            |      Shape |
| ----------------- | ---------: |
| Prediction shape  | (15957, 6) |
| Probability shape | (15957, 6) |

## 9. Validation Results

| Label         | ROC-AUC | Precision | Recall | F1-score | Accuracy |
| ------------- | ------: | --------: | -----: | -------: | -------: |
| toxic         |  0.9441 |    0.9513 | 0.4193 |   0.5820 |   0.9420 |
| severe_toxic  |  0.9401 |    0.2857 | 0.0126 |   0.0241 |   0.9898 |
| obscene       |  0.9450 |    0.9577 | 0.3690 |   0.5328 |   0.9652 |
| threat        |  0.8438 |    0.0000 | 0.0000 |   0.0000 |   0.9977 |
| insult        |  0.9350 |    0.8649 | 0.2779 |   0.4207 |   0.9613 |
| identity_hate |  0.8801 |    1.0000 | 0.0064 |   0.0127 |   0.9903 |

### Average Results

| Metric    |  Score |
| --------- | -----: |
| ROC-AUC   | 0.9147 |
| Precision | 0.6766 |
| Recall    | 0.1809 |
| F1-score  | 0.2621 |
| Accuracy  | 0.9744 |

## 10. Results Interpretation

The baseline model showed good ROC-AUC scores for most labels. This means the model can generally separate toxic and non-toxic comments. The best results were achieved for `toxic`, `obscene`, and `insult` labels because these classes have more examples in the dataset.

However, recall and F1-score were low for rare labels such as `threat`, `severe_toxic`, and `identity_hate`. This happened because the dataset is imbalanced. For example, the `threat` label has only 478 examples, so the model could not learn this class well.

Accuracy is high, but it is not the best metric for this problem because most comments are non-toxic. That is why ROC-AUC, recall, and F1-score are more useful for evaluating this model.

## 11. Challenges

The main challenges in this week were:

* imbalanced label distribution
* rare toxic categories
* noisy online comments
* short comments with limited context
* overlapping toxic labels

These problems can reduce model performance, especially for minority classes.

## 12. Future Improvements

In the next weeks, the project can be improved by using more advanced models and techniques.

Possible improvements:

* apply class balancing methods
* tune model hyperparameters
* use Logistic Regression or Linear SVM as stronger baselines
* use deep learning models such as LSTM or GRU
* use Transformer-based models such as BERT
* improve preprocessing and tokenization

## 13. Conclusion

In Week 2, a complete preprocessing and baseline machine learning pipeline was created for toxic comment classification. The dataset was loaded, cleaned, analyzed, split into train, validation, and test sets, and transformed using TF-IDF. A baseline Naive Bayes model was trained and evaluated.

The model achieved an average ROC-AUC score of **0.9147**, which is a good result for a baseline model. However, low recall and F1-score show that the model still needs improvement, especially for rare toxic categories.

Overall, Week 2 created a strong foundation for future deep learning experiments.

## 14. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Hugging Face Datasets

## 15. References

* Jigsaw Toxic Comment Classification Challenge
* Hugging Face Datasets Documentation
* Scikit-learn Documentation
* Research papers about toxic comment detection
