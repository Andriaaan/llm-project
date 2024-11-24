import streamlit as st
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from litellm import completion
import os
import nltk

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


# --- Завантаження даних ---
@st.cache_data
def load_data():
    # Додаємо параметр trust_remote_code=True
    dataset = load_dataset("pg19", split="train", trust_remote_code=True, streaming=True)
    # Вибираємо перші 20 записів
    dataset_subset = dataset.take(20)  # Використовуємо метод .take() для вибору частини датасету
    chunked_dataset = []

    for record in dataset_subset:  # Тепер працюємо лише з вибраними 20 книгами
        book_chunks = split_into_chunks(record["text"])
        chunked_dataset.extend(book_chunks)

    return chunked_dataset


# Завантажуємо корпус текстів
chunked_dataset = load_data()

# --- Інтерфейс Streamlit ---
st.title("Система Питання-Відповідь на основі Retrieval-Augmented Generation з Groq (Llama3-8b-8192)")
st.write("Ця система знаходить відповіді на ваші запитання, використовуючи сторонні тексти як джерело знань.")

# Введення запиту
query = st.text_input("Введіть запит:")

# Вибір параметрів
top_k = st.slider("Кількість результатів для пошуку BM25:", 1, 10, 5)

# Виконання запиту
if query:
    st.write("### Результати BM25:")

    # Пошук через BM25
    bm25_results = bm25_search(chunked_dataset, query, top_k=top_k)

    # Виводимо результати BM25
    for i, (doc, score) in enumerate(bm25_results, 1):
        st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
        st.write(doc[:300] + "...")  # Показуємо перші 300 символів чанка

    # Об'єднуємо знайдені чанки для моделі
    context = "\n\n".join([doc for doc, _ in bm25_results])

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
