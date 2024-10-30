import numpy as np

class Word2VecExtractor:
    def __init__(self, text_column, vector_size=100, window=5, min_count=1, epochs=10):
        self.text_column = text_column
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.vocabulary = {}
        self.embeddings = {}

    def build_vocab(self, sentences):
        word_freq = defaultdict(int)
        for sentence in sentences:
            words = sentence.split('|')
            for word in words:
                word_freq[word] += 1
        self.vocabulary = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) if freq >= self.min_count}
        self.embeddings = {word: np.random.rand(self.vector_size) for word in self.vocabulary}

    def train(self, sentences):
        # Simplified training: Average word vectors for each sentence
        for epoch in range(self.epochs):
            for sentence in sentences:
                words = sentence.split('|')
                valid_words = [word for word in words if word in self.vocabulary]
                if not valid_words:
                    continue
                for word in valid_words:
                    context = [w for w in valid_words if w != word]
                    if not context:
                        continue
                    for ctx_word in context:
                        # Update embeddings with a simple learning rule
                        self.embeddings[word] += 0.01 * (self.embeddings[ctx_word] - self.embeddings[word])

    def extract_features(self, data):
        sentences = data[self.text_column].tolist()
        self.build_vocab(sentences)
        self.train(sentences)

        feature_matrix = []
        for sentence in sentences:
            words = sentence.split('|')
            valid_words = [word for word in words if word in self.vocabulary]
            if not valid_words:
                feature_matrix.append(np.zeros(self.vector_size))
                continue
            vectors = [self.embeddings[word] for word in valid_words]
            feature_vector = np.mean(vectors, axis=0)
            feature_matrix.append(feature_vector.tolist())

        return feature_matrix