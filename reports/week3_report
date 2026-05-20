# Week 3 Report: Deep Learning Model Training and Experiments

## Project Title

Automated Detection of Cyberbullying and Toxic Behavior in Social Networks

## 1. Overview

In Week 3, the project moved from a traditional machine learning baseline model to a deep learning approach using PyTorch. The main objective of this week was to implement and train an LSTM neural network for multi-label toxic comment classification.

The dataset used in this project is:

`m375zhan/modified_jigsaw_toxic_comment_classification_challenge`

The model predicts the following labels:

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

---

## 2. Work Completed

The following tasks were completed this week:

* loaded and prepared the dataset
* cleaned and normalized text comments
* created train, validation, and test splits
* built a vocabulary from training data
* converted comments into integer sequences
* applied sequence padding and truncation
* implemented a PyTorch Dataset and DataLoader pipeline
* created an LSTM-based neural network
* trained the model using BCEWithLogitsLoss
* evaluated the model using classification metrics
* saved training history and validation results

---

## 3. Text Preprocessing

The comments were cleaned before training the neural network.

The preprocessing steps included:

* converting text to lowercase
* removing URLs
* removing punctuation and numbers
* removing extra spaces
* tokenizing text into words

Example:

```text
Original: YOU are stupid!!! Visit http://test.com
Cleaned: you are stupid
```

---

## 4. Dataset Split

The dataset was divided into three parts:

| Split      |   Size |
| ---------- | -----: |
| Train      | 30,000 |
| Validation |  5,000 |
| Test       | 15,958 |

A smaller training subset was used to reduce training time and make experimentation faster.

---

## 5. Vocabulary and Tokenization

A vocabulary was built from the most common words in the training dataset.

Main settings:

| Parameter               |   Value |
| ----------------------- | ------: |
| Maximum vocabulary size |  30,000 |
| Maximum sequence length |     200 |
| Padding token           | `<PAD>` |
| Unknown token           | `<UNK>` |

Each text comment was converted into a sequence of integer token IDs.

---

## 6. Deep Learning Model

The deep learning model used in Week 3 is an LSTM neural network implemented in PyTorch.

### Model Architecture

```text
Embedding Layer
↓
LSTM Layer
↓
Dropout Layer
↓
Fully Connected Layer
↓
Output Predictions
```

### Hyperparameters

| Parameter             |             Value |
| --------------------- | ----------------: |
| Embedding dimension   |               128 |
| Hidden dimension      |               128 |
| Number of LSTM layers |                 1 |
| Batch size            |                64 |
| Epochs                |                 2 |
| Learning rate         |             0.001 |
| Optimizer             |              Adam |
| Loss function         | BCEWithLogitsLoss |

`BCEWithLogitsLoss` was used because the task is multi-label classification.

---

## 7. Training Process

The model was trained using PyTorch training loops.

During training:

* forward propagation was performed
* loss was calculated
* gradients were computed using backpropagation
* optimizer updated model weights

Training and validation loss values were stored after each epoch.

---

## 8. Evaluation Metrics

The model was evaluated using:

* ROC-AUC
* Precision
* Recall
* F1-score
* Accuracy

These metrics are important because the dataset is imbalanced and accuracy alone is not enough to evaluate performance.

---

## 9. Results

The Week 3 model successfully completed training and generated predictions for all toxic labels.

Main observations:

* the LSTM model can learn sequential patterns in text
* the model understands word order better than TF-IDF
* rare labels are still difficult because the dataset is imbalanced
* deep learning training requires more computation time compared to the baseline model

The training history and validation metrics were saved into the `results/` folder.

---

## 10. Challenges

The main challenges during Week 3 were:

* long training time
* imbalanced toxic labels
* noisy online comments
* GPU limitations
* memory usage during training

To reduce training time, a smaller subset of the training dataset was used.

---

## 11. Future Improvements

In Week 4, the following improvements are planned:

* compare Week 2 baseline and Week 3 LSTM results
* improve recall and F1-score
* apply hyperparameter tuning
* experiment with bidirectional LSTM
* try Transformer-based models such as BERT
* create final visualizations and result tables
* prepare final presentation and report

---

## 12. Conclusion

In Week 3, a complete deep learning pipeline for toxic comment classification was implemented using PyTorch and LSTM networks.

The project now includes:

* preprocessing pipeline
* tokenization and vocabulary creation
* neural network model
* training loop
* evaluation metrics
* saved model checkpoints

This week established the main deep learning component of the project and prepared the system for final improvements in Week 4.
