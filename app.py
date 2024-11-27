import json

import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from litellm import completion
import nltk
import os
from sentence_transformers import SentenceTransformer, util
from rerankers import Reranker
from datasets import load_dataset

# --- Реранкер ---
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

nltk.download('punkt')
nltk.download('punkt_tab')

# --- Ініціалізація API ключа для Groq ---
# Встановлюємо API ключ для Groq
os.environ['GROQ_API_KEY'] = "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"  # Замініть на ваш реальний ключ


# --- Розбиття тексту на чанки ---
def split_into_chunks(text, chunk_size=50):
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


# --- Повторне ранжування через Cross-Encoder ---
def rerank_candidates(candidates, query):
    docs = [doc for doc, _ in candidates]
    query_doc_pairs = [(query, doc) for doc in docs]
    rerank_scores = cross_encoder.predict(query_doc_pairs)
    reranked_results = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return reranked_results


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

@st.cache_data
def load_data_dataset():
    # Завантаження датасету holistic-books
    dataset = load_dataset("0xBreath/holistic-books", split="train")
    # Перевірка значень автора перед фільтрацією
    print("Автори у датасеті:")
    authors = set(record.get("author", "Unknown") for record in dataset)
    print(authors)

    # Фільтрація записів без автора "Walter Last"
    filtered_dataset = dataset.filter(lambda record: record.get("author", "").strip() != "Walter Last")

    # Перевірка кількості записів
    print(f"Кількість книг після фільтрації: {len(filtered_dataset)}")

    for record in filtered_dataset.select(range(min(50, len(filtered_dataset)))):
        title = record.get("source", "Unknown Title")
        author = record.get("author", "Unknown Author")
        print(f"{title} | {author}")

    # Розбиваємо книги на чанки
    chunked_dataset = []
    for record in dataset:
        text = record.get("text", "")
        book_chunks = split_into_chunks(text)
        chunked_dataset.extend(book_chunks)

    return chunked_dataset


def save_data():
    chunked_dataset = load_data()
    with open("chunked_dataset.json", "w", encoding="utf-8") as f:
        json.dump(chunked_dataset, f, ensure_ascii=False, indent=4)

# Завантажуємо корпус текстів
#chunked_dataset = load_data()

#save_data()

# Завантажуємо збережений корпус текстів
with open("chunked_dataset.json", "r", encoding="utf-8") as f:
    chunked_dataset = json.load(f)


# Завантажуємо модель для семантичного пошуку
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# --- Інтерфейс Streamlit ---
st.title("Система Питання-Відповідь на основі Retrieval-Augmented Generation з Groq (Llama3-8b-8192)")
st.write("Ця система знаходить відповіді на ваші запитання, використовуючи сторонні тексти як джерело знань.")

# Введення запиту
query = st.text_input("Введіть запит:")

# Вибір параметрів
top_k_slider = st.slider("Кількість результатів для пошуку BM25:", 1, 10, 5)
top_k = 30

# Вибір методу пошуку
search_method = st.selectbox("Оберіть метод пошуку:", ["BM25", "Семантичний пошук", "Комбінований пошук"])

# Виконання запиту
if query:
    # Виконання запиту
    if query:
        st.write("### Результати:")

        if search_method == "BM25":
            results = bm25_search(chunked_dataset, query, top_k=top_k)

        elif search_method == "Семантичний пошук":
            results = semantic_search(chunked_dataset, query, model, top_k=top_k)

        elif search_method == "Комбінований пошук":
            bm25_results = bm25_search(chunked_dataset, query, top_k=top_k)
            semantic_results = semantic_search(chunked_dataset, query, model, top_k=top_k)

            # Об'єднуємо результати з BM25 та семантичного пошуку
            results = bm25_results + semantic_results
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        for i, (doc, score) in enumerate(results[:top_k_slider], 1):
            st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
            st.write(doc)

    use_reranker = st.checkbox("Використовувати реранкер для покращення результатів")

    if use_reranker:
        results = rerank_candidates(results, query)
        st.write("Результати після реранкінгу:")
    else:
        results = results
        st.write("Результати без реранкінгу:")

    results = results[:top_k_slider]

    # Виводимо результати
    for i, (doc, score) in enumerate(results, 1):
        st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
        st.write(doc)

    context = "\n\n".join([f"Chunk {index + 1}: {doc}" for index, (doc, _) in enumerate(results)])

    st.write("### Відповідь моделі:")

    # Генеруємо відповідь через Groq (за допомогою litellm)
    st.write("### Model's Answer:")
    response = completion(
        model="groq/llama3-8b-8192",
        messages=[
            {"role": "user", "content": query},
            {"role": "system", "content": f"Context:\n{context}"},
            {
                "role": "system",
                "content": (
                    "Use the context to generate an answer. Please cite the sources in brackets "
                    "using the format [Chunk N] where relevant information is used."
                ),
            },
        ],
    )

    # Виводимо відповідь
    st.write(response['choices'][0]['message']['content'])
