# Week 4 Report: Final Evaluation and Web Application

## Project Title

Automated Detection of Cyberbullying and Toxic Behavior in Social Networks

---

# 1. Introduction

The goal of this project is to build a deep learning system that can automatically detect toxic and offensive comments in online platforms.

In Week 4, the main focus was:

- final model evaluation
- threshold tuning
- model comparison
- error analysis
- deployment of the trained model into a Streamlit web application

The project uses the following dataset:

`m375zhan/modified_jigsaw_toxic_comment_classification_challenge`

---

# 2. Objectives of Week 4

The objectives of Week 4 were:

- improve prediction quality
- compare baseline and deep learning models
- analyze model errors
- create a user-friendly web application
- prepare the final project pipeline

---

# 3. Final Model Evaluation

The trained LSTM model from Week 3 was used for final evaluation.

The model predicts probabilities for six toxic categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The following metrics were used:

- ROC-AUC
- Precision
- Recall
- F1-score
- Accuracy

Threshold tuning was performed using different thresholds:

- 0.2
- 0.3
- 0.4
- 0.5

The best threshold was selected based on F1-score.

---

# 4. Model Comparison

Two models were compared during the project:

| Model | Description |
|---|---|
| TF-IDF + Naive Bayes | Baseline machine learning model |
| LSTM Neural Network | Final deep learning model |

The LSTM model achieved better understanding of context and word order compared to the baseline model.

The final model produced more balanced predictions for toxic categories.

---

# 5. Error Analysis

Error analysis was performed to understand incorrect predictions.

Common problems included:

- short comments
- ambiguous language
- rare toxic labels
- class imbalance
- informal online writing style

This analysis helped identify limitations of the model.

---

# 6. Streamlit Web Application

A Streamlit web application was developed for the final project.

The application allows users to:

- enter comments
- analyze toxicity probabilities
- view detected toxic categories
- adjust prediction threshold
- visualize probabilities using charts and progress bars

The web application loads the trained LSTM model directly from the project.

---

# 7. Technologies Used

The following technologies and libraries were used:

- Python
- PyTorch
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Hugging Face Datasets

---

# 8. Results

The project successfully completed:

- dataset analysis
- text preprocessing
- baseline model creation
- LSTM deep learning model training
- threshold tuning
- final evaluation
- error analysis
- deployment into a web application

The final system can automatically analyze online comments and estimate toxicity risk.

---

# 9. Conclusion

Week 4 completed the final stage of the project.

The project demonstrates how deep learning can be applied to toxic comment detection in social networks.

The final Streamlit application provides an interactive interface for real-time toxicity prediction using the trained LSTM model.

The project combines machine learning, deep learning, NLP, and web deployment into one complete pipeline.
