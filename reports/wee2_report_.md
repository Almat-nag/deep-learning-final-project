# Week 2 Progress Report

## Project Title

Automated Detection of Cyberbullying and Toxic Behavior in Social Media Comments

## 1. Work Completed

During Week 2, I worked on data preprocessing and the baseline model.

Completed tasks:

- loaded the modified Jigsaw Toxic Comment Classification dataset from Hugging Face;
- checked missing values and dataset columns;
- cleaned text comments;
- analyzed label distribution;
- split the dataset into train, validation, and test sets;
- trained a baseline model using TF-IDF and Naive Bayes;
- evaluated the model using ROC-AUC, Precision, Recall, F1-score, and Accuracy.

## 2. Dataset

The dataset contains text comments and six toxicity labels:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

This is a multi-label classification task because one comment can belong to several categories at the same time.

## 3. Baseline Model

For the baseline model, I used TF-IDF vectorization with a Naive Bayes classifier.

This model was chosen because it is simple, fast, and useful for comparing with future deep learning models.

## 4. Results

The baseline model successfully generated predictions for all six labels.

The results were saved in the `results` folder.

The model works better on common labels such as toxic, obscene, and insult, but it may perform worse on rare labels such as threat and identity_hate.

## 5. Problems

The main problem is class imbalance. Some toxic labels appear much less often than others.

Another limitation is that the baseline model does not fully understand context, sarcasm, slang, or indirect insults.

## 6. Plan for Next Week

In Week 3, I plan to build a deep learning model using LSTM or GRU.

I will compare the deep learning model with the Week 2 baseline model and analyze whether it improves the results.

## 7. Conclusion

In Week 2, I completed the preprocessing pipeline and built the first baseline model.

The project pipeline now works from dataset loading to model evaluation.
