# Banglish Sentiment Analysis with Smart Normalizer
This project implements a custom "Smart Filter" for Banglish (Bengali-English) text to improve sentiment analysis accuracy.

## 🚀 The Code
```python
# 1. Environment Setup
!pip install phonetic-bangla transformers torch tqdm nltk scikit-learn

import pandas as pd
import nltk
import re
from phoneticbn import bn
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

print("Setting up NLTK English Dictionary...")
nltk.download('words', quiet=True)
from nltk.corpus import words
english_dict = set(words.words())

def smart_normalizer(text):
    raw_words = str(text).split()
    final_sentence = []
    for word in raw_words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        if clean_word in english_dict and len(clean_word) > 2:
            final_sentence.append(word)
        else:
            final_sentence.append(bn(word))
    return " ".join(final_sentence)

# 2. Dataset Processing
FILE_NAME = 'final_kolkata_banglish_dataset.csv'
df = pd.read_csv(FILE_NAME)

df['Naive_Text'] = df['text'].apply(lambda x: bn(str(x)))
df['Smart_Text'] = df['text'].apply(smart_normalizer)

# 3. AI Evaluation
sentiment_ai = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

true_labels, raw_predictions, naive_predictions, smart_predictions = [], [], [], []

for index, row in tqdm(df.iterrows(), total=len(df)):
    label = str(row['sentiment']).lower().strip()
    
    raw_p = sentiment_ai(str(row['text']), truncation=True)[0]['label'].lower().strip()
    naive_p = sentiment_ai(str(row['Naive_Text']), truncation=True)[0]['label'].lower().strip()
    smart_p = sentiment_ai(str(row['Smart_Text']), truncation=True)[0]['label'].lower().strip()

    true_labels.append(label)
    raw_predictions.append(raw_p)
    naive_predictions.append(naive_p)
    smart_predictions.append(smart_p)

# 4. Results
print("=== SMART FILTER METRICS ===")
print(classification_report(true_labels, smart_predictions, digits=4, zero_division=0))
