import nltk
import spacy
import numpy as np
from collections import defaultdict
import pandas as pd
from rdflib import Literal, Namespace, RDF, OWL, Graph
import re

def extract_initial_ontology(abstracts):
    ontology = defaultdict(list)
    
    # Define a regular expression pattern to extract concepts and synonyms
    pattern = r'(\w+) (?:is|are|has) (\w+)'
    
    for abstract in abstracts:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        for match in matches:
            concept, synonym = match
            ontology[concept].append(synonym)
    
    return ontology

def apply_lexicon_manager(ontology, new_terms):
    for concept, synonyms in new_terms.items():
        if concept not in ontology:
            ontology[concept] = []
        ontology[concept].extend(synonyms)

def extract_conceptual_relational_knowledge(sentence, ontology):
    knowledge = []
    for concept, synonyms in ontology.items():
        for word in synonyms:
            if word in sentence:
                knowledge.append((concept, word))
    return knowledge

if __name__ == "__main__":
    ## 2.1; 2.3. Extract features from document, sentence, and word level
    df = pd.read_csv('prepared_df_dump.csv')

    nlp = spacy.load("en_core_web_sm")

    abstracts = df['abstract']

    document_features = []
    sentence_features = []
    word_features = []

    for abstract in abstracts:
        doc = nlp(abstract)

        doc_features = {
            "num_sentences": len(list(doc.sents)),
            "num_words": len(doc),
        }
        document_features.append(doc_features)

        sentence_features_per_abstract = []
        for sent in doc.sents:
            sent_features = {
                "num_tokens": len(sent),
                "sentence_text": sent.text,
            }
            sentence_features_per_abstract.append(sent_features)
        sentence_features.append(sentence_features_per_abstract)

        word_features_per_abstract = []
        for token in doc:
            word_features = {
                "word_text": token.text,
                "word_pos": token.pos_,
                "word_dep": token.dep_,
            }
            word_features_per_abstract.append(word_features)
        word_features.append(word_features_per_abstract)

    # 2.2. Provide dataset based on the word level, but containing upper structure as attributes

    word_level_dataset = []
    for i, abstract in enumerate(abstracts):
        for j, token_features in enumerate(word_features[i]):
            word_level_dataset.append({
                "abstract_text": abstract,
                "sentence_text": sentence_features[i][j]["sentence_text"],
                "word_text": token_features["word_text"],
                "word_pos": token_features["word_pos"],
                "word_dep": token_features["word_dep"],
            })

    word_level_df = pd.DataFrame(word_level_dataset)

    ## 3.1 Based on input-ontology apply current rules on the sentence structures and morphological analysis
    ontology = extract_initial_ontology(abstracts)
    
    extracted_knowledge = []
    for abstract in df['abstract']:
        doc = nlp(abstract)

        for sentence in doc.sents:
            sentence_text = sentence.text
            knowledge = extract_conceptual_relational_knowledge(sentence_text, ontology)
            extracted_knowledge.append((sentence_text, knowledge))

    ontels = {}
    for sentence, knowledge in extracted_knowledge:
        for concept, word in knowledge:
            if concept not in ontels:
                ontels[concept] = set()
            ontels[concept].add(word)

            # Include word-level features into the ontology
            for idx, row in word_level_df.iterrows():
                if row["word_text"] == word:
                    ontels[concept].add(f"{row['word_pos']}_{row['word_dep']}")

    g = Graph()
    onto = Namespace("http://example.org/ontology#")

    for concept, synonyms in ontology.items():
        for synonym in synonyms:
            g.add((onto[concept], onto.hasSynonym, Literal(synonym)))

    g.serialize("ontology_hasti.owl", format="xml")



    constructed_ontology = Graph()
    constructed_ontology.parse("ontology_hasti.owl", format="xml")
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
