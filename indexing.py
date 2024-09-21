# -------------------------------------------------------------------------
# AUTHOR: Gokul Srinath Seetha Ram
# FILENAME: indexing.py
# SPECIFICATION: Calculates document-term matrix for collection.csv dataset
# FOR: CS 5180- Assignment #1
# TIME SPENT: 3 days
# -----------------------------------------------------------*/
import csv
import math
from collections import Counter

# Reading the documents from a CSV file
documents = []
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if there is one
    for row in reader:
        # Assuming the document text is in the first column
        documents.append(row[0])

# Step 1: Conducting stopword removal for pronouns and conjunctions
stopWords = {"i", "she", "he", "they", "her",
             "their", "and", "is", "the", "a", "an"}


def remove_stopwords(doc):
    words = doc.lower().split()
    return [word for word in words if word not in stopWords]


# Step 2: Conducting stemming
stemming = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
    "love": "love"
}


def apply_stemming(words):
    return [stemming.get(word, word) for word in words]


# Step 3: Preparing the documents
processed_documents = []
for doc in documents:
    words = remove_stopwords(doc)
    stemmed_words = apply_stemming(words)
    processed_documents.append(stemmed_words)

print("Final Documents are:", processed_documents)

# Step 4: Identifying the index terms (vocabulary)
terms = ['love', 'cat', 'dog']  # Fixed order of terms

# Step 5: Calculating TF, IDF, and TF-IDF
N = len(processed_documents)

# Function to calculate term frequency (TF)


def compute_tf(term, doc):
    return doc.count(term) / len(doc)  # Normalized term frequency

# Function to calculate inverse document frequency (IDF)


def compute_idf(term):
    doc_count = sum(1 for doc in processed_documents if term in doc)
    return math.log10(N / doc_count) if doc_count > 0 else 0


# Print IDF values for debugging
for term in terms:
    idf = compute_idf(term)
    print(f"IDF for {term}: {idf:.4f}")

# Constructing the TF-IDF matrix
docTermMatrix = []
for doc in processed_documents:
    tfidf_values = []
    for term in terms:
        tf = compute_tf(term, doc)
        idf = compute_idf(term)
        tfidf = tf * idf
        tfidf_values.append(tfidf)
    docTermMatrix.append(tfidf_values)

# Step 6: Printing the document-term matrix
print("------------Document Term Matrix------------")
print(f"{'':<15}{'love':<10}{'cat':<10}{'dog':<10}")  # Term headers
for idx, doc_label in enumerate(['d1', 'd2', 'd3']):
    row = [f"{doc_label:<15}"] + \
        [f"{value:.2f}".ljust(10) for value in docTermMatrix[idx]]
    print("".join(row))
