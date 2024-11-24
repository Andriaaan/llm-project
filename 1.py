import os
import requests

# Створіть директорію для збереження книг, якщо вона не існує
books_directory = "books"
if not os.path.exists(books_directory):
    os.makedirs(books_directory)

# Список URL книг з Project Gutenberg
books = [
    {"title": "Alice's Adventures in Wonderland", "url": "https://www.gutenberg.org/files/11/11-0.txt"},
    {"title": "Moby Dick", "url": "https://www.gutenberg.org/files/2701/2701-0.txt"},
    {"title": "Pride and Prejudice", "url": "https://www.gutenberg.org/files/1342/1342-0.txt"}
]

# Завантаження та збереження кожної книги в задану директорію
for book in books:
    response = requests.get(book["url"])
    file_name = book["title"].replace(" ", "_") + ".txt"
    file_path = os.path.join(books_directory, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Книга '{book['title']}' була завантажена та збережена в файл '{file_path}'")
