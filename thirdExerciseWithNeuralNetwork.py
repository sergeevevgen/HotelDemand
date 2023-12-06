import math
import os

from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns
import joblib

# Загрузим данные
ndf = pd.read_csv("hotel_bookings_raw.csv")
ndf.dropna(inplace=True)


def neural_network_task3(df):
    # Объявляю объект для преобразования строковых значений в числовые
    label_encoder = LabelEncoder()

    # Путь к сохраненной моделе
    model_path = 'static/models/model_neural_task3.keras'

    # Выберем признаки и целевую переменную (доход)
    features_list = ['lead_time', 'stays_in_weekend_nights',
                     'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'customer_type',
                     'previous_cancellations',
                     'previous_bookings_not_canceled', 'required_car_parking_spaces',
                     'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'GDP', 'CPI_HOTELS']

    # Выбираем признаки и целевую переменную
    features = df[features_list].copy()
    target = df['adr'].copy()

    # Применяю к каждому столбцу признака преобразования
    for f in features_list:
        features[f] = label_encoder.fit_transform(features[f])

    # Разделяем данные на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

    # Масштабируем признаки для лучшей производительности модели
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создаем и обучаем MLPRegressor
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    if os.path.isfile(model_path):
        # Загрузка модели
        mlp_model = joblib.load(model_path)
    else:
        mlp_model.fit(X_train_scaled, y_train)

    # Делаем предсказания на тестовых данных
    y_predictions = mlp_model.predict(X_test_scaled)

    # Оцениваем производительность модели
    mse = mean_squared_error(y_test, y_predictions)
    mae = mean_absolute_error(y_test, y_predictions)
    r2 = r2_score(y_test, y_predictions)

    print(f"Среднеквадратичная ошибка (MSE): {round(math.sqrt(mse), 2)}%")
    print(f"Среднеабсолютное отклонение (MAE): {round(mae, 2)}%")
    print(f"Коэффициент детерминации (R^2): {round(r2, 4) * 100}%")

    # График
    plt.figure(figsize=(10, 7))
    plt.plot(y_test.values, label='Фактические', marker='o', color='#8b00ff')
    plt.plot(y_predictions, label='Предсказанные', marker='o', color='#ff294d')
    plt.title('Фактические vs предсказанные значения')
    plt.xlabel('Фактические')
    plt.ylabel('Предсказанные')
    plt.legend(loc='best')
    plt.savefig('static/images/neural_task3.png')
    # plt.show()
    plt.clf()

    # Сохранение модели
    if not os.path.isfile(model_path):
        joblib.dump(value=mlp_model, filename=model_path)


neural_network_task3(ndf)
