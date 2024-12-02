import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["document_index"]
terms_collection = db["terms"]
documents_collection = db["documents"]

# Sample documents (from question 3)
documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

# Preprocess text
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Tokenize into unigrams, bigrams, and trigrams
def tokenize(text):
    words = text.split()
    unigrams = words
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    return unigrams + bigrams + trigrams

# Build the inverted index
def build_inverted_index():
    terms_collection.delete_many({})
    documents_collection.delete_many({})

    for doc_id, content in enumerate(documents, start=1):
        preprocessed_content = preprocess(content)
        tokens = tokenize(preprocessed_content)

        # Add document to MongoDB
        documents_collection.insert_one({"_id": doc_id, "content": content})

        # Count term frequencies
        term_frequencies = {}
        for pos, token in enumerate(tokens):
            if token not in term_frequencies:
                term_frequencies[token] = []
            term_frequencies[token].append(pos)

        # Add terms to inverted index
        for term, positions in term_frequencies.items():
            term_entry = terms_collection.find_one({"term": term})
            if term_entry:
                term_entry["docs"].append({"doc_id": doc_id, "positions": positions})
                terms_collection.replace_one({"term": term}, term_entry)
            else:
                terms_collection.insert_one({
                    "term": term,
                    "docs": [{"doc_id": doc_id, "positions": positions}]
                })

# Query the inverted index and rank documents using cosine similarity
def query_index(queries):
    # Prepare the vectorizer for TF-IDF
    vectorizer = TfidfVectorizer()
    preprocessed_docs = [preprocess(doc) for doc in documents]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    feature_names = vectorizer.get_feature_names_out()

    for query_id, query_text in enumerate(queries, start=1):
        preprocessed_query = preprocess(query_text)
        query_vector = vectorizer.transform([preprocessed_query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Rank documents by similarity score
        ranked_docs = sorted(
            [(doc, score) for doc, score in zip(documents, similarities) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )

        # Display results
        print(f"Query {query_id}: {query_text}")
        for doc, score in ranked_docs:
            print(f"  Document: \"{doc}\", Score: {score:.4f}")
        print()

# Main script
if __name__ == "__main__":
    build_inverted_index()

    # Queries from the question
    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    query_index(queries)
