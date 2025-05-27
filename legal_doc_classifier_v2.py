
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem.porter import PorterStemmer
from nltk import download

download('punkt')

mapping_df = pd.read_csv('Interview_Mapping.csv')
mapping_df['Judgements'] = mapping_df['Judgements'].astype(str) + '.txt'
file_to_label = mapping_df.set_index('Judgements')['Area.of.Law'].to_dict()

judgement_dir = 'Fixed_Judgements'
data = []

for file in os.listdir(judgement_dir):
    if file.endswith(".txt") and file in file_to_label:
        with open(os.path.join(judgement_dir, file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().replace('\n', ' ')
            data.append((text, file_to_label[file]))

df = pd.DataFrame(data, columns=['Judgement', 'Class'])

df_test = df[df['Class'] == 'To be Tested'].copy()
df = df[df['Class'] != 'To be Tested'].copy()

def clean_text(text):
    text = re.sub(r'[<^>]+', ' ', text)
    text = re.sub(r'[\W_]+', ' ', text.lower())
    return text

df['Judgement'] = df['Judgement'].apply(clean_text)
df_test['Judgement'] = df_test['Judgement'].apply(clean_text)

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

X_train, y_train = df['Judgement'].values, df['Class'].values
X_test, y_test = df_test['Judgement'].values, df_test['Class'].values

pipeline = Pipeline([
    ('vect', TfidfVectorizer(strip_accents='unicode', lowercase=True)),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
])

param_grid = {
    'vect__ngram_range': [(1, 2)],
    'vect__stop_words': ['english'],
    'vect__tokenizer': [tokenizer_porter],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
}

start_time = time.time()

grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"CV Accuracy: {grid_search.best_score_:.3f}")

clf = grid_search.best_estimator_

y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Set Accuracy: {test_acc:.3f}")
print(classification_report(y_test, y_pred))
print(f"Total Time Taken: {time.time() - start_time:.2f} seconds")
