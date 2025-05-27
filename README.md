# Legal-Docs-Classifier
**Smart Sorting: Saving Time by Automatically Classifying Legal Documents**
This project classifies legal documents (judgements) by their area of law using machine learning.

- Loads and processes hundreds of legal text files

- Cleans and tokenizes text with stemming (Porter)

- Transforms text using TF-IDF (word frequency importance)

- Trains a logistic regression model with hyperparameter tuning (via grid search)

- Evaluates model performance on a separate test set

**Key Findings**
Best model uses:

- TF-IDF with 1–2 word n-grams

- English stop words removed

- L1 or L2 logistic regression penalties

- Achieved ~63% accuracy on cross-validation


Final test set accuracy depends on test size and label distribution but was comparable to training

The model learned to associate legal phrases with specific legal categories like Constitutional Law, Family Law, or Contract Law
