1. Загрузка и подготовка данных
"""

import pandas as pd
from sklearn.model_selection import train_test_split


url = 'https://drive.google.com/uc?export=download&id=1M7X-wqyQdOiBmjGisiChUtP7S4MRX8Vv'
data = pd.read_csv(url)


X = data['text']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""2. Извлечение признаков"""

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

"""3. Обучение модели"""

from sklearn.linear_model import PassiveAggressiveClassifier


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

"""4. Оценка модели и вывод текстовых результатов"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Фейк', 'Реальная'], output_dict=False)


print(f'Точность: {score*100:.2f}%')
print('\nОтчет о классификации:\n', report)

"""5. Визуализация матрицы ошибок"""

import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Фейк', 'Реальная'], yticklabels=['Фейк', 'Реальная'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')
plt.show()

"""6. Визуализация распределения типов новостей в тестовых данных"""

plt.figure(figsize=(6, 6))
sns.countplot(x=y_test, hue=y_test, palette='Set2', dodge=False)
plt.title('Распределение реальных и фейковых новостей в тестовых данных')
plt.xlabel('Тип новостей')
plt.ylabel('Количество')
plt.show()

"""7. Визуализация точности модели"""

plt.figure(figsize=(6, 6))
labels = ['Правильно классифицировано', 'Неправильно классифицировано']
sizes = [conf_matrix[0][0] + conf_matrix[1][1], conf_matrix[0][1] + conf_matrix[1][0]]
colors = ['#4CAF50', '#FF5252']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Распределение точности')
plt.show()

"""4) Вывод: Представленная модель демонстрирует высокую эффективность в обнаружении фальшивых новостей. Визуализации помогут заказчику глубже понять результаты и принять обоснованные решения.

---

Задание: Обнаружение болезни Паркинсона

Цель: Создать модель машинного обучения, способную с высокой точностью (более 90%) предсказывать наличие болезни Паркинсона на ранней стадии, используя алгоритм XGBoost и библиотеку sklearn для нормализации признаков.

Описание задачи:

1) Данные:

Для обучения и тестирования модели используется датасет, содержащий акустические параметры голосовых записей пациентов. Датасет включает признаки, такие как частота, джиттер и шиммер, а также метку (status), указывающую на наличие (1) или отсутствие (0) болезни Паркинсона.

Ссылка на датасет:https://drive.google.com/file/d/1wn9tAKtl1DUXwolwBx3N0A8CFe1Z7TA6/view?usp=drive_link


Этапы работы:

1) Загрузка и предобработка данных:

* Загрузка данных из предоставленного файла.
* Установка корректных заголовков для столбцов и удаление ненужных столбцов (например, name).
* Разделение данных на признаки (X) и метки (y).
* Разделение данных на обучающую (80%) и тестовую (20%) выборки.

2) Нормализация данных:

* Нормализуйте признаки с помощью StandardScaler из библиотеки sklearn для приведения данных к единому масштабу.

3) Построение и обучение модели:

* Пострение модели классификации с использованием XGBClassifier из библиотеки xgboost.
* Обучение модели на обучающих данных.

4) Оценка качества модели:

* Оценка точности модели на тестовой выборке.
* Построение матрицы ошибок и отчет о классификации для анализа результатов.

5) Визуализация:

* Представление результатов оценки модели в виде графиков и диаграмм, включая визуализацию матрицы ошибок и отчет о классификации.

6) Результаты:

* Точность: Модель достигла точности 95% на тестовых данных.
* Матрица ошибок: Показывает количество правильных и неправильных классификаций для каждого класса.
* Отчет о классификации: Включает показатели точности (precision), полноты (recall) и F1-score для каждого класса.

1. Импорт библиотек и загрузка данных
"""

import pandas as pd
import warnings


warnings.filterwarnings('ignore')


url = 'https://drive.google.com/uc?id=1wn9tAKtl1DUXwolwBx3N0A8CFe1Z7TA6'
df = pd.read_csv(url, header=1)


columns = ['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
           'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
           'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
           'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA', 'spread1',
           'spread2', 'D2', 'PPE']


df.columns = columns


df.head()

"""2. Подготовка данных"""

df = df.drop('name', axis=1)


X = df.drop('status', axis=1)
y = df['status']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""3. Нормализация данных"""

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""4. Обучение модели и оценка"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print(f'Точность модели: {accuracy:.2f}')
print('\nМатрица ошибок:')
print(conf_matrix)
print('\nОтчет о классификации:')
print(class_report)
