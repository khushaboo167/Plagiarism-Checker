import PyPDF2
print(PyPDF2.__version__)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Simulated corpus (replace with real data in production)
corpus = [
    "Machine learning is the study of computer algorithms that improve automatically through experience.",
    "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural human intelligence.",
    "Photosynthesis is the process by which green plants synthesize food from carbon dioxide and sunlight.",
    "The capital of France is Paris, known for its art, fashion, and culture.",
]

# Step 3: Detect plagiarism
def detect_plagiarism_from_pdf(pdf_text, corpus):
    documents = corpus + [pdf_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    max_score = np.max(similarity_scores)
    plagiarism_percent = max_score * 100

    print(f"\nPlagiarism Detected: {plagiarism_percent:.2f}%")
    if plagiarism_percent > 75:
        print("⚠️ High plagiarism detected.")
    elif plagiarism_percent > 40:
        print("⚠️ Moderate plagiarism.")
    else:
        print("✅ Low or no significant plagiarism.")

# Step 4: Run it
pdf_path = input("Enter the path to your PDF file:\n")
pdf_text = extract_text_from_pdf(pdf_path)
detect_plagiarism_from_pdf(pdf_text, corpus)
