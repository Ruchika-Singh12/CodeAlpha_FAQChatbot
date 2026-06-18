# 🤖 AI FAQ Chatbot
### NLP-based Question Answering System using TF-IDF + Cosine Similarity

---

## 📌 Project Overview

This chatbot answers user questions by matching them to the most similar FAQ
using NLP techniques: **TF-IDF Vectorization** and **Cosine Similarity**.

| Technique        | Purpose                                      |
|------------------|----------------------------------------------|
| TF-IDF           | Convert text to numerical vectors            |
| Cosine Similarity| Find the most similar FAQ question           |
| Text Preprocessing | Lowercase, remove punctuation, clean text  |
| Flask            | Backend web server                           |

---

## 📁 Project Structure

```
faq_chatbot/
├── app.py              ← Main Flask backend (NLP logic here)
├── faqs.py             ← FAQ dataset (questions & answers)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── templates/
    └── index.html      ← Chat UI frontend
```

---

## ⚙️ Setup Instructions

### Step 1: Open terminal in VS Code
```
Ctrl + ` (backtick)
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
python app.py
```

### Step 4: Open browser
```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

```
User Question
     ↓
Preprocessing (lowercase, remove punctuation)
     ↓
TF-IDF Vectorization (convert to numbers)
     ↓
Cosine Similarity (compare with all FAQs)
     ↓
Best Match → Display Answer + Confidence Score
```

---

## 💡 Sample Questions to Try

- What is artificial intelligence?
- Explain machine learning
- How does deep learning work?
- What is NLP?
- Tell me about neural networks
- What is Python used for?
- What is cosine similarity?

---

## 📚 Technologies Used

- **Python 3.x**
- **Flask** – Web framework
- **scikit-learn** – TF-IDF & Cosine Similarity
- **NumPy** – Numerical operations
- **HTML/CSS/JavaScript** – Chat UI

---

## Developer
Ruchika Singh

## Internship Organization
CodeAlpha
*Built as Task 2: Chatbot for FAQs — NLP Project*
