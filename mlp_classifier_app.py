import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Заголовок приложения
st.title("MLP Classifier: Пример на данных 'Ирисов'")

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Отображение первых 5 строк данных
st.write("### Первые 5 строк данных")
st.write(iris.data[:5])

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Настройки MLP через Streamlit
st.sidebar.header("Настройки MLP")
hidden_layer_sizes = st.sidebar.slider("Размер скрытого слоя", min_value=10, max_value=100, step=10, value=50)
max_iter = st.sidebar.slider("Максимум итераций", min_value=200, max_value=1000, step=100, value=500)
alpha = st.sidebar.slider("Регуляризация (alpha)", min_value=0.0001, max_value=0.01, step=0.0001, value=0.0001)

# Создание и обучение MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, alpha=alpha, random_state=42)
mlp.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = mlp.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Точность модели: {accuracy * 100:.2f}%")

# Отображение матрицы ошибок
st.write("### Матрица ошибок")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
st.pyplot(fig)

# Отображение графика обучения
st.write("### График обучения")
loss_values = mlp.loss_curve_
plt.figure()
plt.plot(loss_values)
plt.title("Зависимость потерь от итераций")
plt.xlabel("Итерации")
plt.ylabel("Потери")
st.pyplot(plt)
