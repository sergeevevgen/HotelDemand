import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import seaborn as sns

# df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')
# df.dropna(inplace=True)


# Функция для отображения дерева решений
def decision_tree_task1(df):
    # Преобразование типов питания к числам
    # Инициализация LabelEncoder
    label_encoder = LabelEncoder()

    # Путь к сохраненной моделе
    model_path = 'static/models/model_tree_task1.keras'

    # Преобразование столбцов в числовые значения для всего фрейма
    list_params = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children',
        'babies', 'meal',
        'customer_type', 'previous_cancellations', 'previous_bookings_not_canceled',
        'required_car_parking_spaces', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG',
        'GDP', 'CPI_HOTELS']

    df = df.copy()

    for i in list_params:
        df[i] = label_encoder.fit_transform(df[i])

    # Количество элементов для обучения (99%)
    count_to_train = round(len(df) * 0.99)
    # Количество элементов для тестирования (1%)
    count_to_test = len(df) - count_to_train

    # Набор данных для обучения для всего фрейма
    train_df = df.head(count_to_train).copy()

    # Флаг отмены бронирования - целевая переменная
    y = train_df.copy()['is_canceled']

    # Уровень инфляции и потребительские настроения по отношению к экономике для всего фрейма
    x = train_df.copy()[list_params]

    # Создание модели дерева решений
    model = DecisionTreeClassifier()

    if os.path.isfile(model_path):
        # Загрузка модели
        model = joblib.load(model_path)
    else:
        # Обучение модели
        model.fit(x, y)

    # Проверка модели для всего фрейма
    test_df = df.tail(count_to_test).copy()

    y_test = test_df.copy()['is_canceled']
    x_test = test_df.copy()[list_params]

    prediction = model.score(x_test, y_test)

    print('Качество дерева решений: ', prediction * 100, '%')
    if not os.path.isfile('static/images/decision_tree.png'):
        # Визуализация дерева решений
        plt.figure(figsize=(12, 8))
        plot_tree(model, feature_names=list_params, filled=True)

        # Сохранение графика в файл .png
        plt.savefig('static/images/decision_tree.png', dpi=1000)
        plt.clf()

    y_predictions = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_predictions)

    # График для матрицы неточностей
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('static/images/confusion_matrix_tree_task1.png')
    plt.clf()

    # График
    plt.figure(figsize=(10, 7))
    plt.plot(y_predictions.flatten(), label='Предсказанные', marker='o', color='#ff294d')
    plt.plot(y_test.values, label='Фактические', marker='o', color='#8b00ff')
    plt.title('Фактические vs предсказанные значения')
    plt.xlabel('Фактические')
    plt.ylabel('Предсказанные')
    plt.legend(loc='best')
    plt.savefig("static/images/decision_tree_graph.png")
    # plt.show()
    plt.clf()

    res = sorted(dict(zip(list(x.columns), model.feature_importances_)).items(),
                 key=lambda el: el[1], reverse=True)

    print('feature importance:')
    for val in res:
        print(val[0] + " - " + str(val[1] * 100) + '%')

    # Сохранение модели
    if not os.path.isfile(model_path):
        joblib.dump(value=model, filename=model_path)


# tree_visual()
