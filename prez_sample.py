import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def iris_sample():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, alpha=alpha, random_state=42)
    mlp.fit(X_train, y_train)

    # Предсказание на тестовых данных
    y_pred = mlp.predict(X_test)
    print(pd.DataFrame(y_pred))
    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Точность модели: {accuracy * 100:.2f}%")

