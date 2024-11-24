import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from litellm import completion
import nltk
import os
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('punkt_tab')

# --- Ініціалізація API ключа для Groq ---
# Встановлюємо API ключ для Groq
os.environ['GROQ_API_KEY'] = "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"  # Замініть на ваш реальний ключ


# --- Розбиття тексту на чанки ---
def split_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# --- Пошук через BM25 ---
def bm25_search(corpus, query, top_k=5):
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(corpus[i], score) for i, score in top_results]


# --- Семантичний пошук через Sentence-BERT ---
def semantic_search(corpus, query, model, top_k=5):
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cosine_scores = cosine_scores.cpu().detach().numpy()
    top_results = np.argpartition(-cosine_scores, range(top_k))[:top_k]
    return [(corpus[i], cosine_scores[i].item()) for i in top_results]


# --- Завантаження даних з локальної директорії ---
@st.cache_data
def load_data():
    books_directory = r"F:\1\llm\books"  # Ваш шлях до директорії з книгами
    book_texts = []

    # Завантажуємо всі текстові файли з директорії
    for filename in os.listdir(books_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(books_directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                book_texts.append(f.read())

    # Розбиваємо на чанки
    chunked_dataset = []
    for text in book_texts:
        book_chunks = split_into_chunks(text)
        chunked_dataset.extend(book_chunks)

    return chunked_dataset


# Завантажуємо корпус текстів
chunked_dataset = load_data()

# Завантажуємо модель для семантичного пошуку
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# --- Інтерфейс Streamlit ---
st.title("Система Питання-Відповідь на основі Retrieval-Augmented Generation з Groq (Llama3-8b-8192)")
st.write("Ця система знаходить відповіді на ваші запитання, використовуючи сторонні тексти як джерело знань.")

# Введення запиту
query = st.text_input("Введіть запит:")

# Вибір параметрів
top_k = st.slider("Кількість результатів для пошуку BM25:", 1, 10, 5)

# Вибір методу пошуку
search_method = st.selectbox("Оберіть метод пошуку:", ["BM25", "Семантичний пошук", "Комбінований пошук"])


# Виконання запиту
if query:
    # Виконання запиту
    if query:
        st.write("### Результати:")

        if search_method == "BM25":
            bm25_results = bm25_search(chunked_dataset, query, top_k=top_k)
            for i, (doc, score) in enumerate(bm25_results, 1):
                st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
                st.write(doc[:300] + "...")

        elif search_method == "Семантичний пошук":
            semantic_results = semantic_search(chunked_dataset, query, model, top_k=top_k)
            for i, (doc, score) in enumerate(semantic_results, 1):
                st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
                st.write(doc[:300] + "...")

        elif search_method == "Комбінований пошук":
            bm25_results = bm25_search(chunked_dataset, query, top_k=top_k)
            semantic_results = semantic_search(chunked_dataset, query, model, top_k=top_k)

            # Об'єднуємо результати з BM25 та семантичного пошуку
            combined_results = bm25_results + semantic_results
            combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:top_k]

            for i, (doc, score) in enumerate(combined_results, 1):
                st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
                st.write(doc[:300] + "...")

    # Об'єднуємо знайдені чанки для моделі, залежно від вибраного методу пошуку
    if search_method == "BM25":
        context = "\n\n".join([doc for doc, _ in bm25_results])

    elif search_method == "Семантичний пошук":
        context = "\n\n".join([doc for doc, _ in semantic_results])

    elif search_method == "Комбінований пошук":
        # Об'єднуємо результати BM25 та семантичного пошуку
        combined_results = bm25_results + semantic_results
        combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:top_k]
        context = "\n\n".join([doc for doc, _ in combined_results])

    st.write("### Відповідь моделі:")

    # Генеруємо відповідь через Groq (за допомогою litellm)
    response = completion(
        model="groq/llama3-8b-8192",
        messages=[
            {"role": "user", "content": query},
            {"role": "system", "content": f"Context: {context}"}
        ]
    )

    # Виводимо відповідь
    st.write(response['choices'][0]['message']['content'])
