import math
from collections import defaultdict

class TFIDFExtractor:
    def __init__(self, text_column, max_features=1000):
        self.text_column = text_column
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}

    def extract_features(self, data):
        # Calculate Term Frequency (TF)
        tf = []
        document_count = len(data)
        term_document_counts = defaultdict(int)

        for text in data[self.text_column]:
            terms = text.split('|')
            term_freq = defaultdict(int)
            unique_terms = set()
            for term in terms:
                term_freq[term] += 1
                unique_terms.add(term)
            tf.append(term_freq)
            for term in unique_terms:
                term_document_counts[term] += 1

        # Calculate Inverse Document Frequency (IDF)
        for term, count in term_document_counts.items():
            self.idf[term] = math.log((1 + document_count) / (1 + count)) + 1

        # Build Vocabulary
        all_terms = sorted(self.idf.keys(), key=lambda x: self.idf[x], reverse=True)
        self.vocabulary = {term: idx for idx, term in enumerate(all_terms[:self.max_features])}

        # Calculate TF-IDF
        tfidf_matrix = []
        for term_freq in tf:
            vector = [0.0] * len(self.vocabulary)
            for term, freq in term_freq.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    vector[idx] = freq * self.idf[term]
            tfidf_matrix.append(vector)

        return tfidf_matrix