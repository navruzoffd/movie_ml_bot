import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
import pandas as pd
import missingno as msno
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import nltk





nltk.download('punkt')
nltk.download('stopwords')
def process_json(file):
    # Открываем JSON файл для чтения с указанием кодировки
    with open(file, encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Преобразуем данные из JSON в DataFrame
    df = pd.DataFrame(data)

    # Распаковываем столбец 'docs' в отдельные столбцы 'name', 'description' и 'genres'
    df[['name', 'description', 'genres']] = df['docs'].apply(
        lambda x: pd.Series([x['name'], x['description'], ', '.join([genre['name'] for genre in x['genres']])]))

    # Удаляем столбец 'docs', так как мы уже распаковали его содержимое
    df = df.drop(columns=['docs'])

    # Сохраняем данные в CSV файл
    df.to_csv('data.csv', index=False)

    return df

# Получаем путь к файлу examples.json
file_path = 'examples.json'

# Пример использования функции
df = process_json(file_path)


#Предобработка датафрейма
df['description'] = df['description'].str.lower()

df['description'] = df['description'].str.replace('[{}]'.format(string.punctuation), '')

# Токенизация
df['description'] = df['description'].apply(word_tokenize)

# Удаление стоп слов
stop_words_russian = set(stopwords.words('russian'))
df['description'] = df['description'].apply(lambda x: [word for word in x if word not in stop_words_russian])

# Приведение слов к их изначальной форме
morph = pymorphy2.MorphAnalyzer()
df['description'] = df['description'].apply(lambda x: [morph.parse(word)[0].normal_form for word in x])


# Эмбеддинг для описания и жанров
# Создаем список текстовых данных для обучения модели word embeddings и жанров
text_data = df['description'].tolist()
genre_data = [genre.lower().replace(' ', '_').split('|') for genre in df['genres'].tolist()]

# Обучаем модель Word2Vec для описания
description_model = Word2Vec(sentences=text_data, vector_size=100, window=5, min_count=1, workers=4)

# Обучаем модель Word2Vec для жанров
genre_model = Word2Vec(sentences=genre_data, vector_size=50, window=5, min_count=1, workers=4)

# Получаем векторное представление для каждой строки фильма
X = []  # Список для хранения векторов описания
y = []  # Список для хранения векторов жанров

for index, row in df.iterrows():
    description = row['description']  # Описание фильма
    genre = row['genres'].lower().replace(' ', '_').split('|')  # Жанры фильма

    # Получение эмбеддинга для описания
    description_embedding = np.mean(
        [description_model.wv[word] for word in description if word in description_model.wv], axis=0)

    # Получение эмбеддинга для жанра
    genre_embedding = np.mean([genre_model.wv[g] for g in genre if g in genre_model.wv], axis=0)

    if description_embedding is not None and genre_embedding is not None:
        X.append(description_embedding)
        y.append(genre_embedding)

X = np.array(X)
y = np.array(y)

# Размерность X и y
print("Размерность X:", X.shape)
print("Размерность y:", y.shape)



# Создание модели нейронной сети
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))  # Скрытый слой с 128 нейронами
model.add(Dense(y.shape[1], activation='linear'))  # Выходной слой с размерностью эмбеддинга жанра

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(X_train, y_train, epochs=14, batch_size=32, validation_data=(X_test, y_test))


# Оценка модели
loss = model.evaluate(X_test, y_test)

print("Потери на тестовом наборе:", loss)
# Предсказание жанров с помощью обученной модели
predicted_genres = model.predict(X_test)

# Преобразование предсказанных значений в формат жанров (например, one-hot encoding)
predicted_genres_idx = np.argmax(predicted_genres, axis=1)
true_genres_idx = np.argmax(y_test, axis=1)

# Вычисление accuracy - доли правильно угаданных жанров
accuracy = np.mean(predicted_genres_idx == true_genres_idx)
print("Доля правильно предсказанных жанров (accuracy):", accuracy)
'''
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Визуализация потерь на обучающем и валидационном наборах
plt.plot(history.history['loss'], label='Обучающий набор')
plt.plot(history.history['val_loss'], label='Валидационный набор')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('График потерь на обучающем и валидационном наборах')
plt.legend()
plt.show()
'''
# Предобработка сообщения пользователя
# Преобразование в нижний регистр
def preprocess_user_message(user_message: str):
    """
    Предобработка сообщения пользователя.
    """
    print('Введенное сообщение пользователем', user_message)

    # Удаление пунктуации
    user_message = ''.join([char for char in user_message if char not in string.punctuation])

    # Токенизация
    user_message_tokens = word_tokenize(user_message)

    # Удаление стоп-слов
    stop_words_russian = set(stopwords.words('russian'))
    user_message_tokens = [word for word in user_message_tokens if word not in stop_words_russian]

    # Приведение слов к изначальной форме
    morph = pymorphy2.MorphAnalyzer()
    user_message_tokens = [morph.parse(word)[0].normal_form for word in user_message_tokens]

    # Объединение токенов обратно в строку
    processed_user_message = ' '.join(user_message_tokens)

    print("Обработанное сообщение пользователя:", processed_user_message)

    return processed_user_message


def vectorize_user_message(processed_user_message):
    """
    Получение векторного представления сообщения пользователя.
    """
    # Получение векторов слов из модели Word2Vec для каждого слова в обработанном сообщении пользователя
    word_vectors = [description_model.wv[word] for word in processed_user_message.split() if word in description_model.wv]

    # Усреднение векторов слов для получения общего вектора представления сообщения пользователя
    user_message_vector = np.mean(word_vectors, axis=0)

    print("Векторизованное сообщение пользователя:", user_message_vector)

    return user_message_vector


def predict_genre_embedding(user_message_vector):
    """
    Предсказание эмбеддинга жанра из сообщения пользователя с использованием нейронной сети.
    """
    # Предсказание эмбеддинга жанра из сообщения пользователя с использованием нейронной сети
    predicted_genre_embedding = model.predict(user_message_vector.reshape(1, -1))

    # Вывод предсказанного эмбеддинга жанра из сообщения пользователя
    print("Предсказанный эмбеддинг жанра из сообщения пользователя:", predicted_genre_embedding)

    return predicted_genre_embedding


def generate_movies(user_message: str):
    """
    Генерация списка фильмов на основе сообщения пользователя.
    """
    processed_user_message = preprocess_user_message(user_message)
    user_message_vector = vectorize_user_message(processed_user_message)
    predicted_genre_embedding = predict_genre_embedding(user_message_vector)

    # Получение вектора описания из сообщения пользователя
    predicted_description_embedding = np.mean([description_model.wv[word] for word in processed_user_message.split() if word in description_model.wv], axis=0)

    # Вычисление косинусного сходства между эмбеддингами из X и y и эмбеддингами из сообщения пользователя
    genre_similarity = cosine_similarity(predicted_genre_embedding.reshape(1, -1), np.array(y))
    description_similarity = cosine_similarity(predicted_description_embedding.reshape(1, -1), np.array(X))

    # Объединение сходств для получения общего рейтинга фильмов
    overall_similarity = genre_similarity + description_similarity

    # Получение топ-5 наиболее подходящих фильмов
    top_indices = np.argsort(overall_similarity[0])[::-1][:5]
    top_films = df.iloc[top_indices]['name']

    return [film for film in top_films]

'''
# Визуализация векторов
# Объединение всех векторов в один общий массив
combined_vectors_flat = np.array([vec for sublist in combined_vectors for vec in sublist])

# Снижение размерности объединенных векторов до двух компонент с помощью t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=2000, random_state=0)
combined_vectors_tsne = tsne.fit_transform(combined_vectors_flat)

# . M̡i͝s̀s͞ing, [12.04.2024 15:06]
# Визуализация результатов на графике
plt.figure(figsize=(10, 10))
plt.scatter(combined_vectors_tsne[:, 0], combined_vectors_tsne[:, 1], s=10)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Combined Vectors')
plt.show()
'''

'''
# Загрузка сохраненной модели Word2Vec
model = Word2Vec.load('путь_к_вашей_модели_word2vec')
def text_to_vector(text, model):
    words = text.lower().split()  # Приводим текст к нижнему регистру и разбиваем на слова
    word_vectors = [model.wv[word] for word in words if word in model.wv]  # Получаем векторы слов
    if len(word_vectors) > 0:
        text_vector = np.mean(word_vectors, axis=0)  # Усредняем векторы слов
        return text_vector
    else:
        return None

# Пример использования функции
user_input = "комедия новый год"
vector_representation = text_to_vector(user_input, model)

if vector_representation is not None:
    print("Векторное представление введенного текста:")
    print(vector_representation)
else:
    print("Некоторые слова из введенного текста отсутствуют в модели Word2Vec.")

# Пример использования модели для предсказания вектора жанра на основе вектора описания
description_vector = [word_vectors[word] for word in vector_representation.split() if word in word_vectors]
genre_vector_predicted = model.predict([description_vector])

print("Предсказанный вектор жанра:")
print(genre_vector_predicted)
'''