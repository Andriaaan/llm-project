import os
import re

# Директорія, де зберігаються книги
books_directory = r"F:\1\llm\books"

# Зібрати всі текстові файли з директорії
book_texts = []
for filename in os.listdir(books_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(books_directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            book_texts.append(f.read())

# Перевіримо кількість текстів
print(f"Загальна кількість книг: {len(book_texts)}")


# print(f"Текст першої книги:\n{book_texts[0][:500]}")  # Вивести перші 500 символів першої книги

# Тепер у змінній book_texts є список, який містить текст кожної книги.


# Функція для очищення тексту
def clean_text(text):
    text = text.lower()  # Перетворити на малий регістр
    text = re.sub(r"[^a-zа-я0-9\s]", "", text)  # Видалити всі символи, окрім букв, цифр та пробілів
    return text


# Очистимо всі тексти книг
cleaned_texts = [clean_text(text) for text in book_texts]

# Перевіримо очищення
# print(f"Текст після очищення першої книги:\n{cleaned_texts[0][:500]}")  # Вивести перші 500 символів

# Об'єднуємо всі очищені тексти в один великий текст
all_text = " ".join(cleaned_texts)

# Зберігаємо цей текст в окремий файл, якщо потрібно
with open("all_books_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Усі тексти були об'єднані та збережені в файл 'all_books_cleaned.txt'.")
