"""
FAQ Chatbot - AI powered Question Answering System
Uses TF-IDF Vectorization + Cosine Similarity for NLP-based question matching
"""

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
from faqs import faqs

app = Flask(__name__)

# ─────────────────────────────────────────────
# NLP PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text):
    """Clean and normalize text for NLP processing"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# BUILD TF-IDF MATRIX FROM FAQ QUESTIONS
# ─────────────────────────────────────────────

faq_questions = [item["question"] for item in faqs]
faq_answers   = [item["answer"]   for item in faqs]

# Preprocess all FAQ questions
preprocessed_questions = [preprocess_text(q) for q in faq_questions]

# Fit TF-IDF Vectorizer on FAQ questions
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words='english'
)
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)


# ─────────────────────────────────────────────
# MATCHING FUNCTION
# ─────────────────────────────────────────────

def get_best_answer(user_question, threshold=0.15):
    """
    Match user question to most similar FAQ using cosine similarity.
    Returns (answer, matched_question, confidence_score)
    """
    # Preprocess user question
    processed_input = preprocess_text(user_question)

    # Vectorize user question using the fitted vectorizer
    user_vector = vectorizer.transform([processed_input])

    # Compute cosine similarity with all FAQ questions
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Find the best match
    best_idx   = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return {
            "answer":           faq_answers[best_idx],
            "matched_question": faq_questions[best_idx],
            "confidence":       round(float(best_score) * 100, 1),
            "found":            True
        }
    else:
        return {
            "answer": (
                "I'm sorry, I couldn't find a relevant answer to your question. "
                "Please try rephrasing, or ask about AI, Machine Learning, "
                "Deep Learning, NLP, or Python."
            ),
            "matched_question": None,
            "confidence":       round(float(best_score) * 100, 1),
            "found":            False
        }


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json()
    user_message  = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = get_best_answer(user_message)
    return jsonify(result)


@app.route("/faqs", methods=["GET"])
def get_faqs():
    """Return all available FAQ topics"""
    topics = [{"question": q} for q in faq_questions]
    return jsonify(topics)


# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AI FAQ Chatbot Server Starting...")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
