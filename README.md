1. Project Title

Automated Cyberbullying and Toxicity Detection in Social Media

2. Problem Statement

1. What problem are you trying to solve?
The goal of this project is to build an automated content moderation system that detects toxic, abusive, or harassing comments in online forums and social media platforms.
2. Why is this problem useful or interesting?
Cyberbullying is a critical issue that harms the mental health of users and ruins the community experience. Manual moderation cannot scale to handle millions of daily posts. An automated Deep Learning approach provides a fast, scalable layer of protection for users.
3. What will the model predict or generate?
Given a sequence of text (a comment or post), the model will predict the probability that the text contains toxic or abusive language, framing it as a classification task.

3. Dataset

Dataset name: Jigsaw Toxic Comment Classification Challenge

Dataset source link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Number of examples: ~159,000 text samples in the training set.

Input features: Raw user-generated text.

Target labels: Binary categories such as toxic, severe_toxic, obscene, threat, insult, and identity_hate.

Data format: CSV files.

License: Public Domain / Kaggle usage rules.

4. Planned Method

Baseline: Naive Bayes classifier combined with TF-IDF vectorization. This is a classic and robust probabilistic approach for text classification.

Deep learning model: A Recurrent Neural Network (LSTM or GRU) utilizing pre-trained static word embeddings (such as GloVe or Word2Vec) to capture sequential context and semantic relationships in sentences.

Loss function: Binary Cross-Entropy (BCE).

Evaluation metrics: ROC-AUC will be the primary metric, alongside Precision and Recall. Maximizing Recall is highly prioritized here to minimize the number of false negatives (missed toxic comments).

Train/validation/test split plan: 80% Train, 10% Validation, 10% Test.

5. Expected Challenges

Context and Sarcasm: The model might struggle with sarcasm or culturally specific slang that isn't inherently toxic but is used aggressively.

Class Imbalance: The vast majority of comments in the dataset are non-toxic, requiring careful handling (e.g., class weighting) so the model doesn't just predict the majority class.

Out-of-Vocabulary (OOV) Words: Internet users frequently misspell words or invent new abbreviations, which might not be present in pre-trained GloVe embeddings.

6. Weekly Plan

Week	Planned Work	Expected Output
Week 1	Finalize dataset, set up the GitHub repository, and perform Exploratory Data Analysis (EDA) on text lengths and label distributions.	Proposal, initial README, and EDA notebook committed to GitHub.
Week 2	Text preprocessing (removing stopwords, punctuation), data splitting, and building the TF-IDF + Naive Bayes baseline.	Baseline ROC-AUC and Recall results, Week 2 report.
Week 3	Loading GloVe embeddings, implementing the LSTM architecture in PyTorch, and starting the training process.	Trained LSTM weights, loss curves, and Week 3 report.
Week 4	Model tuning, threshold adjustment for Recall optimization, and writing the final documentation.	Final codebase, comprehensive final report, and presentation slides.
