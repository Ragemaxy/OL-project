import pandas as pd
import json
import logging
import dask.bag as db
from typing import Generator, List, Tuple, Optional, Any
import spacy
import nltk
import itertools
from nltk.corpus import stopwords
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)


PATH_ = "arx/arxiv-metadata-oai-snapshot.json"
NUM_PAPERS = 5000

def get_dataset_generator(path: str) -> Generator:
    with open(path, "r") as fp:
        for line in fp:
            row = json.loads(line)
            yield row


def create_dataframe(generator: Generator) -> pd.DataFrame:
    titles = []
    authors = []

    abstracts = []
    categories = []
    dates = []

    for row in generator:
        if len(abstracts) == NUM_PAPERS:
            break

        titles.append(row["title"])
        authors.append(row["authors"])

        dates.append(row["update_date"])
        abstracts.append(row["abstract"])
        categories.append(row["categories"])

    return pd.DataFrame.from_dict({
        "title": titles,
        "authors": authors,
        "date": dates,
        "abstract": abstracts,
        "categories": categories
    })


dataset_generator = get_dataset_generator(
    path=PATH_
)

dataset_df = create_dataframe(dataset_generator)
dataset_df["date"] = pd.to_datetime(dataset_df["date"])


## 1.1 Stopword filtering using spaCy and NLTK stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
df['abstract'] = df['abstract'].apply(lambda text: " ".join([word for word in text.split() if word.lower() not in stop_words]))

## 1.2 POS tagging using huggingface pretrained POS tagger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.do_word_tokenize = True
model = AutoModelForTokenClassification.from_pretrained(model_name)
pos_tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, device=device)
df['abstract_tokens'] = df['abstract'].apply(lambda text: pos_tagger(text))
df['word'] = df['abstract_tokens'].apply(lambda tokens: [token['word'] for token in tokens])
df['pos_token'] = df['abstract_tokens'].apply(lambda tokens: [token['entity'] for token in tokens])
df = df.explode('word').explode('pos_token')
df.reset_index(drop=True, inplace=True)

## 1.3. Frequency analysis 
all_words = [word for tokens in df['abstract_tokens'] for word in tokens]
fdist = FreqDist(all_words)


## 1.4. Key terms extraction
df['word_frequencies'] = df.groupby('word')['word'].transform('count')
whole_dataset_word_counts = Counter([word for text in df['abstract'] for word in text.split()])
df['word_frequencies_in_dataset'] = df['word'].map(whole_dataset_word_counts)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['abstract'])
df['tfidf'] = [tfidf_matrix[i].toarray()[0] for i in range(len(df))]
df['keywords'] = df.apply(lambda row: [word for word, tfidf_score in zip(row['word'].split(), row['tfidf']) if tfidf_score > 0.5], axis=1)
df.reset_index(drop=True, inplace=True)
df.to_csv('prepared_df_dump.csv')
