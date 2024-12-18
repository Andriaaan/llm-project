# llm-project

# Система Питання-Відповідь з використанням BM25, Семантичного Пошуку та Groq (Llama3-8b-8192)

Цей проект реалізує систему Питання-Відповідь (QA), яка використовує методи пошуку BM25, семантичний пошук за допомогою BERT і генерацію відповідей через модель Groq. Модель Llama3-8b-8192 використовується для генерації відповідей на основі контексту, який надається за допомогою методів пошуку.

## Опис

Проект дозволяє:
1. Використовувати пошук за допомогою BM25 для знаходження найбільш релевантних чанків тексту.
2. Використовувати семантичний пошук на основі BERT для знаходження схожих за змістом частин тексту.
3. Інтегрувати модель Groq для генерації відповідей на запитання на основі знайденого контексту.
4. Можливість вмикати або вимикати різні типи пошуку (BM25, семантичний пошук, або комбіноване використання).

## Технології

- **Streamlit** – для побудови веб-інтерфейсу.
- **RankBM25** – для пошуку через алгоритм BM25.
- **SBERT (Sentence-BERT)** – для семантичного пошуку.
- **Groq API** – для генерації відповідей за допомогою моделі Llama3-8b-8192.
- **NLTK** – для токенізації тексту.


## Встановлення

1. Клонувати репозиторій:

   ```bash
   git clone https://github.com/Andriaaan/llm-project.git
   cd llm-project
   
2. Створіть віртуальне оточення та активуйте його:

   ```bash
   python -m venv venv
   source venv/bin/activate  # на Linux/MacOS
   venv\Scripts\activate     # на Windows
   
3. Встановіть залежності:

   ```bash
   pip install -r requirements.txt

## Запуск

Щоб запустити проект, виконайте наступні кроки:

1. Запустіть додаток за допомогою Streamlit:

   ```bash
    streamlit run app.py

2. Після виконання цієї команди у терміналі ви побачите посилання на локальний сервер (зазвичай це http://localhost:8501), де ви зможете використовувати веб-інтерфейс для введення запитань.

3. Веб-сторінка відкриється автоматично в браузері, де ви зможете:

  * Вводити запитання.
  * Вибирати метод пошуку (BM25, семантичний пошук або комбінований пошук).
  * Отримувати відповіді на основі текстів, що зберігаються в системі.


