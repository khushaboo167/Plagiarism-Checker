from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Simulated database of documents (corpus)
corpus = [
    "Machine learning is the study of computer algorithms that improve automatically through experience.",
    "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.",
    "Deep learning is a subset of machine learning concerned with algorithms inspired by the structure and function of the brain.",
    "The capital of France is Paris. It is known for its art, fashion, and culture.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water."
]

# Function to detect plagiarism
def detect_plagiarism(input_paragraph, corpus):
    documents = corpus + [input_paragraph]  # add user paragraph to corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between input and all corpus docs
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Max similarity score as plagiarism percentage
    max_score = np.max(similarity_scores)
    plagiarism_percent = max_score * 100

    print(f"\nPlagiarism Detected: {plagiarism_percent:.2f}%")
    if plagiarism_percent > 75:
        print("⚠️ High plagiarism detected.")
    elif plagiarism_percent > 40:
        print("⚠️ Moderate plagiarism.")
    else:
        print("✅ Low or no significant plagiarism.")

# Input from user
user_input = input("Enter your paragraph:\n")
detect_plagiarism(user_input, corpus)
