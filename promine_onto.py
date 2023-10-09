import nltk
from nltk.corpus import wordnet
import spacy
import numpy as np
from collections import Counter, defaultdict
from data_prep import *
from rdflib import Literal, Namespace, RDF, RDFS, OWL
from sklearn.metrics import precision_recall_fscore_support

## 2.1. Extract synonyms using WordNet and Wiktionary
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())

    return synonyms


## 2.2. Calculating Relations with context similarity between concepts
def calculate_similarity(concept1, concept2):
    doc1 = nlp(concept1)
    doc2 = nlp(concept2)
    similarity = doc1.similarity(doc2)

    return similarity


## 2.3. Extract various pre-versions of Domain Corpuses based on different thresholds
def extract_corpus_with_threshold(data, threshold):
    corpus = defaultdict(list)

    for index, row in data.iterrows():
        abstract = row['abstract']
        keywords = row['keywords']

        for keyword in keywords:
            synonyms = get_synonyms(keyword)

            for syn in synonyms:
                similarity = calculate_similarity(keyword, syn)

                # Extract concepts based on threshold similarity
                if similarity >= threshold:
                    corpus[keyword].append(abstract)

    return corpus


## 3.1-2 Calculate entropy and information gain between different pre versions of Domain Corpuses and choose most appropriate Corpus and Calculate information gain during feeding different documents to ensure viability of the current corpus

def calculate_entropy(corpus):
    total_documents = sum(len(docs) for docs in corpus.values())
    entropy = 0.0

    for concept, docs in corpus.items():
        probability = len(docs) / total_documents
        entropy -= probability * np.log2(probability)

    return entropy

def calculate_information_gain(corpus_before, corpus_after):
    entropy_before = calculate_entropy(corpus_before)
    entropy_after = calculate_entropy(corpus_after)
    information_gain = entropy_before - entropy_after
    return information_gain


def calculate_all_information_gains(corpuses):
    all_information_gains = []
    permutations = itertools.permutations(corpuses, 2)

    for corpus1, corpus2 in permutations:
        information_gain = calculate_information_gain(corpus1, corpus2)
        all_information_gains.append((corpus1, corpus2, information_gain))

    return all_information_gains


## 4.1. Based on hierarchical clustering evaluate different concept groups

def hierarchical_clustering(corpus, num_clusters):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(corpus)
    return cluster_labels

## 4.2. Calculate most related words to this concept hierarchy level
def calculate_related_words(cluster_label, corpus, threshold):
    related_words = []

    for word, docs in corpus.items():
        word_cluster_labels = [G.nodes[i]['cluster'] for i in range(len(corpus[word]))]

        if cluster_label in word_cluster_labels:
            cluster_count = word_cluster_labels.count(cluster_label)
            total_count = len(word_cluster_labels)
            cluster_proportion = cluster_count / total_count

            if cluster_proportion >= threshold:
                related_words.append(word)

    return related_words


if __name__ == "__main__":
  corpuses = []
  for i in range(10):
    threshold = i/10
    corpuses.append(extract_corpus_with_threshold(df, threshold))


  all_information_gains = np.array(calculate_all_information_gains(corpuses))
  corpus_mvp = corpuses[np.argmax(all_information_gains,axis=0)]

  G = nx.Graph()

  num_clusters = 5
  cluster_labels = hierarchical_clustering(corpus_mvp, num_clusters)

  for i, label in enumerate(cluster_labels):
      G.add_node(i, cluster=label)

  for i, label in enumerate(cluster_labels):
      threshold = 0.5
      related_words = calculate_related_words(label, corpus, threshold)
      related_dictionary[label] = related_words

  onto = Namespace("http://example.org/ontology#")

  for node_id, data in G.nodes(data=True):
      concept_label = f"Concept_{node_id}" 
      g.add((onto[concept_label], RDF.type, onto.Concept))
      for related_word in data['related_words']:
          g.add((onto[concept_label], onto.hasRelatedWord, Literal(related_word)))

  g.serialize("promine_ontology2809.owl", format="xml")  

  constructed_ontology = Graph()
  constructed_ontology.parse("promine_ontology2809.owl", format="xml")
  golden_truth_ontology = Graph()
  golden_truth_ontology.parse("gt_wiki_onto.owl", format="xml")

  constructed_onto = "http://example.org/ontology#"
  golden_truth_onto = "http://example.org/gt#"

  golden_truth_elements, constructed_elements = [], []

  for s, p, o in constructed_ontology:
      if str(s).startswith(constructed_onto):
          constructed_elements.append(str(s))

  for s, p, o in golden_truth_ontology:
      if str(s).startswith(golden_truth_onto):
          golden_truth_elements.append(str(s))

  tp = len(set(constructed_elements).intersection(golden_truth_elements))
  fp = len(constructed_elements) - tp
  fn = len(golden_truth_elements) - tp

  accuracy = tp / (tp + fp + fn)
  completeness = tp / (tp + fn)
  f1_score = 2 * (accuracy * completeness) / (accuracy + completeness)

  print(f"Accuracy: {accuracy:.2f}")
  print(f"Completeness (Recall): {completeness:.2f}")
  print(f"F1-Score: {f1_score:.2f}")
