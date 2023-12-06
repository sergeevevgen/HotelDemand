import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')
# df.dropna(inplace=True)


def neural_network_task1(df):
    # Преобразование типов питания к числам
    # Инициализация LabelEncoder
    label_encoder = LabelEncoder()

    # Преобразование столбцов в числовые значения для всего фрейма
    list_params = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children',
        'babies', 'meal',
        'customer_type', 'previous_cancellations', 'previous_bookings_not_canceled',
        'required_car_parking_spaces', 'CPI_AVG', 'INFLATION', 'INFLATION_CHG',
        'GDP', 'CPI_HOTELS']

    for i in list_params:
        df[i] = label_encoder.fit_transform(df[i])

    # Количество элементов для обучения (99%)
    count_to_train = round(len(df) * 0.99)
    # Количество элементов для тестирования (1%)
    count_to_test = len(df) - count_to_train

    # Набор данных для обучения для всего фрейма
    train_df = df.head(count_to_train).copy()

    # Флаг отмены бронирования - целевая переменная - тренировочные данные
    y = train_df.copy()['is_canceled']
    x = train_df.copy()[list_params]

    # Данные для тестирования
    test_df = df.tail(count_to_test).copy()
    y_test = test_df.copy()['is_canceled']
    x_test = test_df.copy()[list_params]

    # Стандартизация данных
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_test = scaler.fit_transform(x_test)

    # Построение модели нейронной сети
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(x, y, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    # Предсказание на тестовых данных
    y_pred = model.predict(x_test)

    # Применение порога к вероятностям
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)

    # Оценка производительности модели с использованием бинарных предсказаний
    accuracy = accuracy_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    print(f'Accuracy: {accuracy * 100}%')
    print('Confusion Matrix:')
    print(conf_matrix)
