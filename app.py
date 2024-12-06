import json
import numpy as np
import streamlit as st
import torch
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from litellm import completion
import nltk
import os
from sentence_transformers import SentenceTransformer, util , CrossEncoder

# Завантаження необхідних ресурсів
nltk.download('punkt')
nltk.download('punkt_tab')

# Ініціалізація моделей
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --- Функції для роботи з ембеддингами ---
def save_embeddings(corpus, embeddings_file="embeddings.npy"):
    corpus_embeddings = model.encode(corpus, convert_to_tensor=False)
    np.save(embeddings_file, corpus_embeddings)
    print("Embeddings saved to file.")


def load_embeddings(embeddings_file="embeddings.npy"):
    if os.path.exists(embeddings_file):
        return np.load(embeddings_file)
    else:
        return None


def build_and_save_bm25_index(chunked_dataset, index_file="bm25_index.pkl"):
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in chunked_dataset]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(index_file, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_index(index_file="bm25_index.pkl"):
    with open(index_file, "rb") as f:
        bm25 = pickle.load(f)
    return bm25


# --- Розбиття тексту на чанки ---
def split_into_chunks(text, chunk_size=50):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# --- Завантаження даних ---
@st.cache_data
def load_data():
    books_directory = r"F:\1\llm\books"
    book_texts = []

    for filename in os.listdir(books_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(books_directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                book_texts.append(f.read())

    chunked_dataset = []
    for text in book_texts:
        book_chunks = split_into_chunks(text)
        chunked_dataset.extend(book_chunks)

    return chunked_dataset


def save_data():
    chunked_dataset = load_data()
    with open("chunked_dataset.json", "w", encoding="utf-8") as f:
        json.dump(chunked_dataset, f, ensure_ascii=False, indent=4)


# --- Семантичний пошук ---
def semantic_search(query, corpus_embeddings, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    corpus_embeddings = torch.tensor(corpus_embeddings).to(device)
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cosine_scores = cosine_scores.cpu().detach().numpy()
    top_results = np.argpartition(-cosine_scores, range(top_k))[:top_k]
    return [(chunked_dataset[i], cosine_scores[i].item()) for i in top_results]


# --- BM25 пошук ---
def bm25_search(query, bm25, corpus, top_k=5):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(corpus[i], score) for i, score in top_results]


# --- Реранкінг ---
def rerank_candidates(candidates, query):
    docs = [doc for doc, _ in candidates]
    query_doc_pairs = [(query, doc) for doc in docs]
    rerank_scores = cross_encoder.predict(query_doc_pairs)
    reranked_results = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return reranked_results


# --- Інтерфейс Streamlit ---
st.title("Система Питання-Відповідь на основі Retrieval-Augmented Generation з Groq (Llama3-8b-8192)")

# --- Введення API ключа ---
api_key = st.text_input("Введіть ваш API ключ:", type="password")
if api_key:
    os.environ['GROQ_API_KEY'] = api_key

    # --- Завантаження даних та ембеддингів ---
    #chunked_dataset = load_data()

    #save_data()

    with open("chunked_dataset.json", "r", encoding="utf-8") as f:
        chunked_dataset = json.load(f)

    # --- Перевірка наявності індексу та його створення ---
    index_file = "bm25_index.pkl"
    if not os.path.exists(index_file):
        st.write("Створюється індекс BM25...")
        chunked_dataset = load_data()
        build_and_save_bm25_index(chunked_dataset, index_file)
        st.write("Індекс BM25 створено та збережено.")
    else:
        st.write("Індекс BM25 завантажується...")
        bm25 = load_bm25_index(index_file)
        st.write("Індекс BM25 завантажено.")

    embeddings = load_embeddings()

    if embeddings is None:
        st.write("Обчислюємо ембеддинги...")
        save_embeddings(chunked_dataset)
        embeddings = load_embeddings()
    else:
        st.write("Ембеддинги завантажено.")

    # --- Введення запиту ---
    query = st.text_input("Введіть запит:")

    # --- Налаштування пошуку ---
    search_method = st.selectbox("Оберіть метод пошуку:", ["BM25", "Семантичний пошук", "Комбінований пошук"])
    top_k_slider = st.slider("Кількість результатів для пошуку:", 1, 10, 5)
    use_reranker = st.checkbox("Використовувати реранкер для покращення результатів")

    # --- Кнопка для запуску пошуку ---
    if st.button("Розпочати пошук") and query:
        st.write("### Результати:")

        # Виконання пошуку
        if search_method == "BM25":
            results = bm25_search(query, bm25, chunked_dataset, top_k=top_k_slider)
        elif search_method == "Семантичний пошук":
            results = semantic_search(query, embeddings, top_k=top_k_slider)
        else:  # Комбінований пошук
            bm25_results = bm25_search(query, bm25, chunked_dataset, top_k=top_k_slider)
            semantic_results = semantic_search(query, embeddings, top_k=top_k_slider)
            results = bm25_results + semantic_results
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k_slider]

        # Реранкінг результатів
        if use_reranker:
            results = rerank_candidates(results, query)
            st.write("Результати після реранкінгу:")

        for i, (doc, score) in enumerate(results, 1):
            st.write(f"**Чанк {i}** (релевантність: {score:.2f})")
            st.write(doc)

        # Формування контексту для відповіді моделі
        context = "\n\n".join([f"Chunk {index + 1}: {doc}" for index, (doc, _) in enumerate(results)])

        # Генерація відповіді
        st.write("### Відповідь моделі:")
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

        st.write(response['choices'][0]['message']['content'])
