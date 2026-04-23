from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.corpus = [d["input"] for d in data]
        self.vectors = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        examples = ""
        for idx in top_indices:
            examples += f"Q: {self.data[idx]['input']}\nA: {self.data[idx]['output']}\n\n"

        return examples