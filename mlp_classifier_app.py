import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from messages import description_msg, theory_msg, pairplot_msg, boxplot_msg
import pandas as pd


# Заголовок приложения
st.title("MLP Classifier: Пример на данных 'Ирисов'")

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# Многостраничная структура
st.sidebar.title("Навигация")

page = st.sidebar.selectbox("Выберите страницу", ["Описание задачи", "Данные", "Анализ данных", "Теория по MLP"])

if page == "Описание задачи":
    st.subheader("Формализация задачи")
    st.markdown(description_msg)
elif page == "Теория по MLP":
    st.subheader("Теория")
    st.markdown(theory_msg)
elif page == "Данные":
    # Загрузка данных
    # Отображение первых 5 строк данных
    st.subheader("Первые 5 строк данных")
    st.write(iris.data[:5], columns=iris.feature_names)

    st.write(class_names[y[:5]])

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

    st.write("### Отчет о классификации")
    report = classification_report(y_test, y_pred, target_names=class_names)
    st.text(report)

    # Визуализация точности на тестовой и тренировочной выборках
    st.write("### График точности на тестовой и тренировочной выборках")

    train_accuracies = []
    test_accuracies = []

    # Можем реализовать процесс частичного обучения с partial_fit, чтобы отслеживать метрику по ходу обучения
    for i in range(1, max_iter + 1, 100):
        mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=i, alpha=alpha, random_state=42, warm_start=True)
        mlp.fit(X_train, y_train)
        
        train_accuracies.append(mlp.score(X_train, y_train))
        test_accuracies.append(mlp.score(X_test, y_test))

    plt.figure()
    plt.plot(range(1, max_iter + 1, 100), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, max_iter + 1, 100), test_accuracies, label='Test Accuracy')
    plt.title('График точности на тестовой и тренировочной выборках')
    plt.xlabel('Итерации')
    plt.ylabel('Точность')
    plt.legend()
    st.pyplot(plt)
elif page == "Анализ данных":
    st.subheader("Первичный анализ данных")
    
    # Корреляционная матрица и тепловая карта
    st.markdown("#### Корреляционная матрица")
    corr_matrix = df.iloc[:, :-1].corr()  # Матрица корреляции только для признаков

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Boxplot для анализа распределений признаков

    st.markdown("#### Boxplot признаков")
    st.markdown(boxplot_msg)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df.iloc[:, :-1], orient="h", palette="Set2", ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(x="species", y="sepal length (cm)", data=df, ax=ax[0, 0])
    sns.boxplot(x="species", y="sepal width (cm)", data=df, ax=ax[0, 1])
    sns.boxplot(x="species", y="petal length (cm)", data=df, ax=ax[1, 0])
    sns.boxplot(x="species", y="petal width (cm)", data=df, ax=ax[1, 1])
    st.pyplot(fig)


    # Pairplot для визуализации взаимосвязей признаков
    st.markdown("#### Pairplot для признаков и классов")
    fig = sns.pairplot(df, hue="species", palette="Set1", corner=True)
    st.pyplot(fig)
    st.markdown(pairplot_msg)